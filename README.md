# 🤖 PLN — Atividade Extra 2.1: Chatbot Clássico de Intenções (`TF-IDF + LinearSVC`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aj1no/PLN_Atividade_Extra_2_1_Chatbot_Classico/blob/main/Aula09_Chatbot_Classico.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF%20%2B%20LinearSVC-orange.svg)
![Dataset](https://img.shields.io/badge/Dataset-BANKING77-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

Repositório referente à **Atividade Extra 2.1 (Aula 09 — Chatbots Clássicos com PLN e Machine Learning)** da disciplina de **Processamento de Linguagem Natural (PLN)**.

---

## 📋 Informações Acadêmicas

- **Curso:** Ciência de Dados (5º Semestre)
- **Disciplina:** Processamento de Linguagem Natural (PLN)
- **Professor:** Leonardo de Lellis Rossi
- **Aluno:** Rodolfo Vinicius Cima Takemoto
- **Atividade:** Notebook Extra 2.1 — Chatbot de Domínio Fechado com Modelo Clássico (`TF-IDF + LinearSVC`)

---

## 🎯 Objetivos do Projeto

1. **Explicar** como um chatbot de domínio fechado reconhece intenções como um problema de classificação supervisionada multiclasse ($f: X \to Y$).
2. **Treinar e avaliar** um classificador de intenções com **Scikit-learn** (`TfidfVectorizer` + `LinearSVC`) no dataset **BANKING77** (10.003 exemplos de treino/validação, 3.080 de teste oficial e 77 intenções bancárias).
3. **Conectar previsões** a respostas predefinidas e calibrar um mecanismo de **fallback** simples baseado na margem de decisão (`decision_function`) no conjunto de validação.
4. **Investigar erros e limitações** da abordagem clássica em três exercícios práticos e responder à pergunta de fechamento sobre a diferença entre métricas offline do classificador e sucesso real de uma conversa.

---

## 📊 Principais Resultados

### 1. Desempenho do Classificador (Exercício 1: Unigramas vs. Unigramas + Bigramas)

A seleção de representação foi conduzida estritamente no **conjunto de validação** (`1.501` mensagens), evitando vazamento de dados (*data leakage*) do conjunto de teste oficial (`3.080` mensagens):

| Modelo | Vocabulário (termos) | Accuracy (Val) | Precision macro (Val) | Recall macro (Val) | F1 macro (Val) | Treino (s) | Erros na Validação (de 1.501) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF (1,2) + LinearSVC` | 21.595 | 0,8894 | 0,8980 | 0,8910 | 0,8910 | 2,613 s | 166 |
| **`TF-IDF (1,1) + LinearSVC`** | **2.135** | **0,8961** | **0,9012** | **0,8961** | **0,8955** | **2,235 s** | **156** |

- **Conclusão do Exercício 1:** A representação apenas com **unigramas `(1,1)`** obteve melhor Accuracy (**+0,67 p.p.**), maior F1 macro (**+0,45 p.p.**) e menor tempo de treino com um vocabulário **10x menor**. A adição de todos os bigramas sem filtro de frequência mínima (`min_df=1`) introduz mais de 19.400 termos raros, aumentando a esparsidade e diluindo o peso dos unigramas-chave sob a normalização L2 do TF-IDF.
- No conjunto de **teste oficial** (`3.080` amostras), o pipeline baseline `(1,2)` alcançou **Accuracy = 88,6%** e **F1 macro = 88,6%** (`0,042 ms/mensagem`).

---

### 2. Fallback Fora do Domínio — OOD (Exercício 2)

Com o limiar calibrado na validação em `FALLBACK_THRESHOLD = -0,555` (para atingir $\ge 90\%$ de acurácia nos aceitos com $97,5\%$ de cobertura), avaliamos **8 mensagens fora do domínio bancário**:

- **Rejeitadas pelo limiar de margem (`decision_score < -0,555`):** **3 de 8 (37,5%)** (*dinossauros*, *física quântica* e *Copa do Mundo*).
- **Aceitas pelo limiar de margem (`decision_score >= -0,555`):** **5 de 8 (62,5%)**, das quais **3 mensagens (37,5%)** receberam respostas bancárias ativas (`fallback = False`) por compartilharem termos cotidianos com o vocabulário bancário:
  - *"My car engine is not working after the oil change."* $\to$ `card_not_working` (margem `+0,0487`)
  - *"I lost my library book and my train ticket yesterday."* $\to$ `lost_or_stolen_card` (margem `+0,0768`)
  - *"When is the arrival of the next flight to London?"* $\to$ `card_arrival` (margem `-0,4731`)
- **Por que margem alta $\neq$ intenção correta:** O `LinearSVC` opera sob a **suposição de mundo fechado** (*closed-world assumption*), medindo apenas a distância algébrica aos hiperplanos *One-vs-Rest* das 77 classes bancárias. Palavras fora do vocabulário (*library*, *train*, *engine*, *flight*) são descartadas pelo `TfidfVectorizer` e a normalização L2 concentra todo o peso nos termos conhecidos (*lost*, *not working*, *arrival*).

---

### 3. Análise de Erros (Exercício 3)

Três casos representativos do conjunto de teste foram investigados:

1. **ID 2503 (`"can I have a non virtual card?"`)** — Correta: `order_physical_card` | Prevista: `getting_virtual_card`.
   - *Causa:* Ausência de composicionalidade semântica na negação (`"non virtual"`). Os termos `'virtual'` e `'virtual card'` dominam o produto escalar linear.
   - *Solução proposta:* *Data augmentation* com paráfrases negativas/contrastivas, inclusão de trigramas `(1,3)` ou uso de modelos contextuais (**BERT**).
2. **ID 3005 (`"What do you need for my identity check?"`)** — Correta: `verify_my_identity` | Prevista: `why_verify_identity`.
   - *Causa:* Alta sobreposição lexical (`'identity check'`) e granularidade excessiva entre intenções irmãs (*o que é necessário* vs. *por que é necessário*).
   - *Solução proposta:* Unificação das intenções no desenho do diálogo ou destaque de *features* interrogativas posicionais (`What` vs. `Why`).
3. **ID 2048 (`"how long dies it take for transfers to reflect..."`)** — Correta: `transfer_timing` | Prevista: `balance_not_updated_after_bank_transfer`.
   - *Causa:* Erro de digitação (`'dies'` em vez de `'does'`) que quebra os bigramas `'long does'` e `'does it'`, somado à ambiguidade com reclamações de saldo não atualizado.
   - *Solução proposta:* Uso combinado de **n-gramas de caracteres** (`analyzer='char_wb'`, `ngram_range=(3,5)`) via `FeatureUnion` e perguntas de desambiguação quando as duas maiores margens forem próximas.

---

## 📁 Estrutura do Repositório

```text
PLN_Atividade_Extra_2_1_Chatbot_Classico/
├── Aula09_Chatbot_Classico.ipynb   # Notebook completo com código, saídas e análises
├── README.md                       # Documentação do projeto e síntese dos resultados
├── LICENSE                         # Licença MIT
└── .gitignore                      # Arquivos ignorados pelo Git
```

---

## 🚀 Como Executar

1. Clique no botão **Open in Colab** no topo deste README para abrir o notebook diretamente no Google Colab.
2. Ou execute localmente em um ambiente com Python 3.10+:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   jupyter notebook Aula09_Chatbot_Classico.ipynb
   ```

---

## 🛠️ Relato de Uso de Ferramentas de IA

Conforme exigido nas instruções da disciplina:

- **Ferramenta:** ChatGPT — *Utilizado para melhorar escrita e fluídez do texto.*
- **Ferramenta:** Gemini — *Utilizado para desenvolvimento de código e estruturação da análise dos exercícios.*
