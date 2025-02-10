import cv2
import os
import smtplib
from fpdf import FPDF
from email.message import EmailMessage
from ultralytics import YOLO

# Configurações do e-mail
EMAIL_SENDER =      "grupo44.hackatonfiap@gmail.com" # Seu e-mail
EMAIL_PASSWORD =    "iqqh ljbs agma oadv"            # Senha do app SMTP
EMAIL_RECEIVER =    "fagn2013@gmail.com"             # Destinatário do e-mail

def format_timestamp(seconds):
    """
    Converte o timestamp de segundos para o formato MM:SS.
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_pdf_with_images(image_paths, timestamps, pdf_filename):
    """
    Cria um PDF com 6 imagens por página (2 colunas x 4 linhas) e timestamps formatados como MM:SS.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_font("Arial", size=10)

    images_per_page = 8          # 2 colunas x 3 linhas
    img_w, img_h = 95, 50        # Tamanho das imagens no PDF
    x_positions = [10, 110]  # Posições X para 2 colunas
    y_positions = [20, 80, 140, 200]  # Posições Y para 3 linhas

    for i, img_path in enumerate(image_paths):
        if i % images_per_page == 0:  # Nova página a cada 6 imagens
            pdf.add_page()
            pdf.cell(200, 10, "Relatório de Detecção YOLO", ln=True, align='C')
            pdf.ln(5)

        # Coordenadas para posicionar a imagem
        x = x_positions[(i % images_per_page) % 2]  # Alterna entre 2 colunas
        y = y_positions[(i // 2) % 4] # Alterna entre 4 linhas (dividindo por 2 para 2 colunas)

        # Adicionar a imagem
        pdf.image(img_path, x=x, y=y, w=img_w, h=img_h)

        # Adicionar o timestamp formatado abaixo da imagem
        timestamp_text = f"Tempo: {format_timestamp(timestamps[i])}"
        pdf.text(x, y + img_h + 5, timestamp_text)

    pdf.output(pdf_filename)
    print(f"Relatório PDF gerado: {pdf_filename}")

def send_email_with_pdf(pdf_path):
    """
    Envia um único e-mail contendo o relatório PDF como anexo.
    """
    msg = EmailMessage()
    msg["Subject"] = "Relatório de Detecção YOLO"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("Segue o relatório de detecção em anexo.")

    # Anexar o PDF ao e-mail
    with open(pdf_path, "rb") as pdf_file:
        pdf_data = pdf_file.read()
        msg.add_attachment(pdf_data, maintype="application", subtype="pdf", filename=os.path.basename(pdf_path))

    # Enviar o e-mail
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"E-mail enviado com o relatório: {pdf_path}")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")

def create_numbered_folder(base_folder="detected_frames"):
    counter = 0
    while os.path.exists(f"{base_folder}_{counter:02d}"):
        counter += 1
    output_folder = f"{base_folder}_{counter:02d}"
    os.makedirs(output_folder)
    return output_folder

# Carregar o modelo YOLO treinado
model_trained = YOLO("runs/detect/train/weights/best.pt")

# Abrir o vídeo
video_path = "myTests/video.mp4"
cap = cv2.VideoCapture(video_path)

# Criar diretório para salvar os frames detectados
output_folder = create_numbered_folder()

frame_count = 0
last_detections = {}  # Armazena as últimas detecções para evitar repetição
detected_frames = []  # Lista para armazenar os frames detectados
timestamps = []  # Lista para armazenar os timestamps dos frames detectados

# Obter FPS do vídeo para calcular tempo
fps = cap.get(cv2.CAP_PROP_FPS)

# Processar os frames do vídeo
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Fim do vídeo

    # Calcular timestamp do frame atual
    timestamp = frame_count / fps

    # Rodar inferência do YOLO no frame
    results = model_trained(frame, conf=0.4)

    # Gerar a imagem com as caixas desenhadas
    annotated_frame = results[0].plot()

    # Guardar as detecções do frame atual
    current_detections = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])  # ID da classe detectada
        confidence = float(box.conf[0])  # Confiança da detecção

        # Salvar apenas se a confiança for maior ou for uma nova classe
        if class_id not in last_detections or confidence > last_detections[class_id]:
            current_detections[class_id] = confidence

    # Se houve mudanças nas detecções, salvar o frame e adicioná-lo à lista
    if current_detections:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)
        print(f"Frame salvo: {frame_filename}")

        # Adicionar o frame salvo e seu timestamp às listas
        detected_frames.append(frame_filename)
        timestamps.append(timestamp)

        # Atualizar as últimas detecções
        last_detections = current_detections.copy()

    # # Exibir a imagem anotada
    # cv2.imshow("YOLO Inference", annotated_frame)

    # # Pressionar 'q' para sair
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

    frame_count += 1

# Fechar vídeo e janelas
cap.release()
cv2.destroyAllWindows()

# Criar e enviar o relatório PDF ao final do vídeo
if detected_frames:
    pdf_filename = os.path.join(output_folder, "relatorio_detectado.pdf")
    create_pdf_with_images(detected_frames, timestamps, pdf_filename)
    send_email_with_pdf(pdf_filename)
else:
    print("Nenhum objeto detectado. PDF não gerado.")
