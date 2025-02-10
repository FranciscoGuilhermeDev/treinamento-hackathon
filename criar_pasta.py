import os

def create_numbered_folder(base_folder="detected_frames"):
    """
    Cria uma pasta numerada automaticamente, como 'detected_frames_00', 'detected_frames_01', etc.
    
    Parâmetros:
        base_folder (str): Nome base da pasta.
    
    Retorna:
        str: Caminho da pasta criada.
    """
    counter = 0

    # Encontrar um nome disponível
    while os.path.exists(f"{base_folder}_{counter:02d}"):
        counter += 1

    output_folder = f"{base_folder}_{counter:02d}"
    os.makedirs(output_folder)
    
    return output_folder  # Retorna o nome da pasta criada
