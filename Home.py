import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Pole Detection App",
    page_icon="⚡️",  # Emoji de poste de eletricidade com energia
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cor principal
primary_color = "#0D47A1"

# Personalização da barra lateral
st.sidebar.title("📖 Sobre")
st.sidebar.info(
    """
    Este aplicativo realiza a detecção de postes de eletricidade em imagens, 
    utilizando técnicas de aprendizado de máquina para identificar diferentes classes de tensão. 
    A identificação pode ser de baixa, média ou alta tensão.
    """
)
logo = "./imgs/logo.jpg"  # Certifique-se de que o caminho do logo está correto
# Removi o parâmetro `use_container_width` para compatibilidade com a versão atual do Streamlit
st.sidebar.image(logo)

# Título da página
st.markdown(f"<h1 style='text-align: center; color: {primary_color};'>⚡️ Pole Detection App</h1>", unsafe_allow_html=True)

# Descrição do aplicativo
st.markdown(
    f"""
    <div style="
        background-color: #F0F4F8; 
        padding: 20px; 
        border-radius: 10px; 
        border: 0px solid {primary_color}; 
        box-shadow: 0px 0px 0px rgba(0, 0, 0, 0.1);
        color: #4A4A4A;
        font-size: 18px;
        line-height: 1.6;
        text-align: justify;">
        
        O Pole Detection App utiliza um modelo avançado de aprendizado de máquina para detectar postes de eletricidade em imagens.
        Ele é capaz de identificar postes de baixa, média e alta tensão. Além disso, o aplicativo pode extrair informações de GPS
        contidas nas imagens, caso essas informações estejam disponíveis.
    </div>
    """, 
    unsafe_allow_html=True
)

# Divisor estilizado para separar seções
st.markdown(f"<hr style='border: 1px solid {primary_color}; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Instruções
st.markdown(f"<h2 style='color: {primary_color}; margin-bottom: 15px;'>📋 Instruções</h2>", unsafe_allow_html=True)

markdown = """
<ol style='font-size: 16px; line-height: 1.8; color: #333333;'>
    <li><b>Carregue suas imagens</b>: Utilize a seção de upload para enviar suas imagens. Os formatos suportados são PNG, JPG, JPEG e WEBP.</li>
    <li><b>Ajuste o limite de confiança</b>: Use o controle deslizante para definir o nível de confiança desejado na detecção dos postes.</li>
    <li><b>Visualize os resultados</b>: As imagens processadas e suas respectivas coordenadas GPS (se disponíveis) serão exibidas após o processamento.</li>
    <li><b>Salve as imagens corrigidas</b>: Se a detecção estiver incorreta, você pode corrigir a classe e salvar a imagem com a nova classificação.</li>
    <li><b>Treine o modelo</b>: É possível re-treinar o modelo com novas imagens que você tenha classificado corretamente.</li>
    <li><b>Teste pela Webcam</b>: Utilize a funcionalidade de detecção pela webcam para testar o modelo em tempo real. (Funcionalidade em desenvolvimento)</li>
</ol>
"""

st.markdown(markdown, unsafe_allow_html=True)

# Exemplo de imagem
st.markdown(f"<h2 style='text-align: center; color: {primary_color}; margin-top: 40px;'>🔍 Exemplo de Detecção</h2>", unsafe_allow_html=True)

# Imagens lado a lado: original e com detecção
col1, col2 = st.columns(2)
with col1:
    # Carregar a imagem
    image_path = "./imgs/Low_tension_pole_3 (2).jpg"
    image = Image.open(image_path)

    # Rotacionar a imagem 90 graus no sentido horário
    rotated_image = image.rotate(-90, expand=True)

    # Exibir a imagem rotacionada no Streamlit
    st.image(rotated_image, caption="Imagem Original", use_column_width=True)
with col2:
    st.image("./imgs/Low_tension_pole_3.jpg", caption="Imagem com Detecção", use_column_width=True)

st.markdown(f"<div style='text-align: center; font-style: italic;'>Comparação entre a imagem original e a imagem com detecção de poste.</div>", unsafe_allow_html=True)

# Divisor estilizado para separar seções
st.markdown(f"<hr style='border: 1px solid {primary_color}; margin-top: 40px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Recursos do aplicativo
st.markdown(f"<h2 style='color: {primary_color}; margin-bottom: 15px;'>🔧 Recursos do Aplicativo</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <ul style='font-size: 16px; line-height: 1.8; color: #333333;'>
        <li>Detecção de postes de eletricidade em imagens.</li>
        <li>Extração de coordenadas GPS das imagens (se disponível).</li>
        <li>Download das imagens processadas após a detecção.</li>
        <li>Correção manual da classificação de postes e possibilidade de treinar o modelo com novas imagens.</li>
        <li>Teste em tempo real com detecção via webcam (funcionalidade futura).</li>
    </ul>
    """, 
    unsafe_allow_html=True
)

# Divisor estilizado para separar seções
st.markdown(f"<hr style='border: 1px solid {primary_color}; margin-top: 40px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Link para o modelo Roboflow
st.markdown(f"<h2 style='color: {primary_color}; margin-bottom: 15px;'>🎥 Teste o Modelo pela Webcam</h2>", unsafe_allow_html=True)
st.markdown(
    f"<a style='color: {primary_color}; font-size: 18px; text-decoration: none; font-weight: bold;' href='https://demo.roboflow.com/pt-2ua0w/1?publishable_key=rf_N9byBSZvx6POYi2oBOuUPx9fYRB3' target='_blank'>🔗 Clique aqui para testar o modelo pela webcam</a>", 
    unsafe_allow_html=True
)

# Divisor estilizado para separar seções
st.markdown(f"<hr style='border: 1px solid {primary_color}; margin-top: 40px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Contato ou suporte
st.markdown(f"<h2 style='color: {primary_color}; margin-bottom: 15px;'>📞 Suporte</h2>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="font-size: 16px; line-height: 1.8; color: #333333;">
    Se precisar de ajuda ou tiver dúvidas sobre o funcionamento do aplicativo, entre em contato com a nossa equipe. <br><br>
    📧 <b>E-mail</b>: <a style='color: {primary_color}; font-weight: bold; text-decoration: none;' href='mailto:poledetectionteams@gmail.com'>poledetectionteams@gmail.com</a>
    </div>
    """, 
    unsafe_allow_html=True
)
