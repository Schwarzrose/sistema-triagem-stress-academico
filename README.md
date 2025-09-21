#Machine Learning para Análise de Stress Acadêmico 

## Sobre o Projeto
Um fator determinante para o desempenho de um indivíduo, em praticamente qualquer tarefa que deve ser executada, depende do seu estado mental. O estresse ignorado pode, além de prejudicar a qualidade de vida, acarretar condições psicológicas mais agravantes.

O conjunto de dados utilizado neste projeto foi retirado da plataforma **Kaggle** e é composto por 140 registros. Estes registros contêm respostas para perguntas que variam desde o ambiente no qual o estudante está inserido até a percepção individual do nível de estresse que sentia no momento. O público-alvo da pesquisa foi composto por estudantes do ensino médio, graduação e pós-graduação. Os dados foram coletados por meio de uma pesquisa informal utilizando o _Google Forms_.

O projeto tem como objetivo construir, aplicando técnicas de **aprendizado de máquina**, um sistema integrado de suporte, capaz de _classificar_ o nível de estresse dos estudantes em três categorias (baixo, moderado e alto) e _prever_ um índice (em um intervalo de 1 a 5) de estresse, a fim de auxiliar na tarefa de encaminhamento ou comunicação da instituição de ensino com o estudante.

No Brasil, a lei determina como obrigatório a presença de atendimento psicológico nas instituições públicas de ensino. Essa prática, apesar de não obrigatória, frequentemente é estendida para as universidades, que disponibilizam seus próprios canais de suporte. 

A relevância do projeto se dá pela possibilidade de **facilitar a interação entre instituição e estudante.** A instituição tem a possibilidade de colher resultados com pesquisas oficiais periodicamente e, após as duas etapas do modelo, um profissional humano realiza o julgamento final com base nos dados para definir o melhor curso de ação.

## Objetivos
- Construir um modelo **classificador** para categorizar os níveis de estresse.
- Construir um modelo **regressor** para prever o índice de estresse
- Construir um modelo **clusterizador** para agrupar os estudantes estressados em grupos específicos.
- Construir um sistema integrado combinando o classificador e o regressor para auxiliar na identificação de casos que precisam de suporte psicológico.

## Tecnologias Utilizadas
- Python 3.13+
- Jupyter Notebook
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Matplotlib
- Seaborn

## Metodologia
O projeto foi dividido em quatro etapas principais, cada uma com seu respectivo notebook contendo informações detalhadas sobre os processos até a conclusão. 

Os algoritmos utilizados durante a primeira etapa do projeto, na construção do classificador, foram o **Random Forest Classifier**, **Logistic Regression** e o **GaussianNB**. Também foram aplicadas estratégias para balanceamento de classes via **SMOTE** e métricas para facilitar a compreensão dos dados, como **confusion_matrix**.

Para a segunda etapa do projeto foram utilizados os seguintes algoritmos: **Random Forest Regressor**, **Linear Regression**, **Ridge**, **ElasticNet** e **SVR**. Os dados foram normalizados via **StandardScaler** e métricas como **R²** e **RMSE** foram usadas para aferir a performance do modelo. 

Durante a construção de ambos modelos as técnicas **cross_val_score** e **GridSearchCV** foram utilizadas para validação e encontrar os melhores hiperparâmetros.

Na terceira etapa do projeto, para a construção dos clusters o principal algoritmo utilizado foi o **KMeans**. O **StandardScaler** para pré processamento e **silhouette_score** para métricas. 

Por fim na quarta, e última, etapa um script **Python** foi escrito para simular um protótipo modesto do sistema integrado em produção. Os algoritmos utilizados foram os que obtiveram a melhor performance nas etapas anteriores: **Random Forest Classifier** e **SVR**, os novos dados utilizados para demonstrar o funcionamento do modelo foram escritos dentro do script, _são fictícios e meramente demonstrativos_.

## Resultados
- Classificador: 71% accuracy
- Regressor: R² 44% -> 84% após o aumento dos registros nos dados via **SMOTE**
- Clustering: Falhou. Silhouette Score de 0.274 com os melhores hiperparâmetros. As variáveis têm correlações muito fracas e não formam grupos naturais.
- Sistema Integrado: Funcional. Consistente dentro das limitações dos modelos, tem uma tendência à classificar todo caso como "estresse Alto".

## Como Executar

- Baixe todos os arquivos do repositório
- Abra os notebooks na ordem das etapas, em qualquer ambiente que suporte `.ipynb` (os notebooks foram escritos no Jupyter através do Anaconda)
- Execute as células sequencialmente (Shift + Enter)
- Para rodar o sistema integrado: execute `python sistema_integrado.py`

⚠️ **Observação:** execute na ordem indicada para garantir dependências entre etapas

## 📝 Conclusões
A escolha desse dataset foi deliberada, uma das principais funções do cientista de dados é resolver problemas de negócio, algo que tende a ser naturalmente desafiador. O dataset em questão, além de possuir pouquíssimos registros, também pertencia a uma área muito complexa: o psicológico do ser humano é repleto de nuances e subjetividade.

A técnica a ser aplicada para resolver um problema ou construir um modelo, naturalmente, depende do objetivo da análise. Minha intenção principal durante esse projeto era conseguir construir modelos capazes de enxergar um padrão nos dados e auxiliar de maneira eficiente na triagem de estudantes que possam precisar de apoio psicológico.

Entretanto, a matemática não se dobra aos caprichos do cientista; no máximo se transforma, mas jamais perde a essência. Com apenas 140 registros, estava nítido desde a conclusão da primeira etapa que o volume de dados seria insuficiente para proporcionar um desempenho adequado. Ao fim da segunda etapa do projeto, através do experimento com os dados sintéticos, foi possível confirmar que o modelo de regressão poderia funcionar muito melhor com uma quantidade adequada de registros.

O terceiro projeto revelou que, às vezes, não importa a quantidade de registros se as variáveis não possuírem uma forte correlação entre si. Não há como determinar grupos específicos se as informações não revelarem características necessárias para o agrupamento.

Enfim, a construção do sistema integrado, dentro das limitações específicas dos dois modelos que compõem o sistema, contribuiu com um vislumbre de um projeto ambicioso que, diante dos obstáculos supracitados, ainda depende muito do fator humano. Naturalmente, a decisão nunca foi excluir o fator humano, principalmente em uma situação que requer atenção psicológica, entretanto, apesar de a maior taxa de confiabilidade do sistema estar na identificação de casos de "estresse alto", ele ainda tem uma tendência em categorizar outras situações como "Alto", forçando o responsável por conferir os resultados a ter atenção redobrada.

De todo modo, foi uma experiência de aprendizado satisfatoriamente desafiadora ou desafiadoramente satisfatória.

## 👨‍💻 Autor
**Vitor Fernandes**  
[LinkedIn](https://www.linkedin.com/in/vitor-fernandes-s/)  
Email: vitorbfernandes.s@gmail.com
