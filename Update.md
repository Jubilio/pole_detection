# 🚧 Atualização do Projeto de Detecção de Postes Elétricos 🚧

Olá, equipe!

Aqui está um resumo das fases do nosso projeto de detecção de postes elétricos (baixa, média e alta tensão) até o momento:

## 📌 Fases Completas

1. **Coleta de Dados**: Já coletamos um conjunto inicial de imagens de postes elétricos para treinar nosso modelo.
2. **Configuração do Ambiente**: Nosso ambiente está preparado tanto no **GitHub** quanto no **Google Colab**.
   - **Código no GitHub**: [Repositório Pole Detection](https://github.com/Jubilio/pole_detection)
   - **Notebook no Colab**: [Google Colab](https://colab.research.google.com/drive/1nsvJBjV4OeLVJmnzTnYN1X0K0GuCgjZ5#scrollTo=I5S4IUx1Wt7b)
3. **Primeiros Testes do Modelo**: Utilizamos o YOLOv8 para começar o treinamento inicial com as imagens que coletamos até agora.

## 🔜 Próximos Passos

1. **Coleta Adicional de Imagens**: Precisamos aumentar o volume de imagens para melhorar a precisão e abrangência do modelo.
   - Ferramenta: [Roboflow](https://app.roboflow.com/jubilio-mausse-eiawh/pole_tension/1)
2. **Re-treinar o Modelo**: Após expandir nosso dataset, voltaremos a treinar o modelo YOLO para aumentar a eficácia da detecção em todas as categorias de tensão.
