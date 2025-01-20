from PIL import Image
import os

# Caminho da pasta contendo as imagens
input_folder = "zed_cam/raw_images"
output_folder_left = "zed_cam/images_calibration/left"
output_folder_right = "zed_cam/images_calibration/right"

# Criar as pastas de saída se não existirem
os.makedirs(output_folder_left, exist_ok=True)
os.makedirs(output_folder_right, exist_ok=True)

# Listar todos os arquivos na pasta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # Processar apenas arquivos .png
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # Remover o canal alfa, se presente
        if image.mode == "RGBA":
            image = image.convert("RGB")

        width, height = image.size

        # Definir as metades esquerda e direita
        left_half = (0, 0, width // 2, height)
        right_half = (width // 2, 0, width, height)

        left_image = image.crop(left_half)
        right_image = image.crop(right_half)

        # Gerar nomes de arquivos para as imagens separadas
        base_filename = os.path.splitext(filename)[0]
        left_image_path = os.path.join(
            output_folder_left, f"{base_filename}L.jpg"
        )
        right_image_path = os.path.join(
            output_folder_right, f"{base_filename}R.jpg"
        )

        # Salvar as imagens
        left_image.save(left_image_path)
        right_image.save(right_image_path)

        print(f"Imagem {filename} dividida e salva com sucesso.")