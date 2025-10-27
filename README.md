<!-- antes de enviar a versão final, solicitamos que todos os comentários, colocados para orientação ao aluno, sejam removidos do arquivo -->
# CLIP BIMASTER

#### Aluno: [Abdigal Camargo](https://github.com/Abdigal1)
#### Orientadora: [Manoela Kohler](https://github.com/manoelakohler)

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

<!-- para os links a seguir, caso os arquivos estejam no mesmo repositório que este README, não há necessidade de incluir o link completo: basta incluir o nome do arquivo, com extensão, que o GitHub completa o link corretamente -->
- [Link para o código](https://github.com/Abdigal1/CLIP_BIMASTER) <!-- caso não aplicável, remover esta linha -->

- Trabalhos relacionados: <!-- caso não aplicável, remover estas linhas -->
    - [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
    - [Lithology Identification in Carbonate Thin Section Images of the Brazilian Pre-Salt Reservoirs](https://ieeexplore.ieee.org/abstract/document/10814886).


---

### Resumo

<!-- trocar o texto abaixo pelo resumo do trabalho, em português -->

A caracterização de rochas em um poço é um processo demorado e suscetível a vieses decorrentes da experiência e do julgamento dos especialistas. No entanto, esse procedimento é fundamental para avaliar a qualidade do poço e determinar seu potencial de extração. Em particular, no contexto do ambiente do pré-sal, a utilização de métodos que possibilitem uma caracterização mais rápida, automatizada e livre de vieses torna-se extremamente valiosa. Ainda mais relevante é dispor de um sistema capaz de gerar descrições detalhadas das rochas em linguagem natural. Este é precisamente o objetivo deste trabalho: utilizar um modelo denominado CLIP e adaptá-lo a uma base de dados com conhecimento técnico especializado, alinhado a lâminas de imagens de rochas.

### Abstract <!-- Opcional! Caso não aplicável, remover esta seção -->

The characterization of rocks within a well is a time-consuming process and is often subject to biases arising from the expertise and judgment of specialists. Nevertheless, this procedure is essential for assessing the quality of the well and determining its extraction potential. In particular, within the pre-salt environment, the development of methods that enable faster, automated, and bias-free characterization becomes highly valuable. Even more significant is the ability to have a system capable of generating detailed, human-readable descriptions of rock samples. This is precisely the objective of this work: to employ a model known as CLIP and adapt it to a database containing specialized technical knowledge, aligned with rock thin-section images.

### 1. Introdução
A caracterização de rochas em poços é uma etapa essencial para avaliar a qualidade e o potencial de extração, porém costuma ser um processo demorado e sujeito a vieses humanos. No contexto do pré-sal, torna-se especialmente relevante desenvolver métodos automatizados e imparciais que acelerem essa análise e aprimorem sua precisão.

Neste trabalho, propõe-se o uso de um modelo multimodal baseado no CLIP, adaptado a uma base de dados composta por imagens de lâminas de rochas e descrições textuais técnicas. O objetivo é explorar a capacidade dessa arquitetura em associar informações visuais e textuais, demonstrando seu potencial para apoiar a caracterização automatizada de rochas carbonáticas no ambiente do pré-sal.

### 2. Modelagem

Os dados recebidos possuem o seguinte formato: um par de imagens correspondentes à mesma lâmina, porém obtidas sob diferentes condições de iluminação. Associadas a essas imagens, cada amostra contém uma descrição, que pode variar desde a simples indicação da classe da rocha até uma caracterização mais detalhada, incluindo a classe, a microporosidade e o processo diagenético em que se encontra.

Esse conjunto de dados é dividido em conjuntos de train e test para a realização de uma validação do tipo zero-shot no conjunto de teste. Os dados de treinamento são processados por arquiteturas baseadas no modelo CLIP, que combina um encoder de texto e um encoder de imagem. Diferentes arquiteturas e modelos pré-treinados são utilizados: no caso do texto, emprega-se o modelo BERTimbau pré-treinado, e para as imagens são testados os encoders EVA, ViT e ViT-Res.

Todos os modelos são treinados por, no mínimo, 20 épocas, com uma paciência de 5 épocas, utilizando diferentes learning rate schedulers para otimização do desempenho.
<img width="847" height="536" alt="image" src="https://github.com/user-attachments/assets/205a33a9-8885-4e4f-9d48-225168ecf013" />


### 3. Resultados

Os resultados e a capacidade do modelo de verificar se foi ajustado ao novo domínio são avaliados por meio de uma classificação zero-shot. Os resultados do melhor modelo — uma combinação do encoder EVA e do BERTimbau — são os seguintes:
![a](https://github.com/user-attachments/assets/11a71df3-6cda-45e9-8003-c9d7cc06a4d3)

### 4. Conclusões

Este modelo demonstra a capacidade de arquiteturas como o CLIP de integrar informações visuais e textuais em um contexto específico, como o da caracterização de lâminas de rochas. Este trabalho representa um ponto de partida para o desenvolvimento de modelos multimodais aplicados à classificação de rochas carbonáticas no ambiente do pré-sal.

---

Matrícula: 222.100.371

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
