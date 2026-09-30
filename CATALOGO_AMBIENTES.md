# BenchTWM: catálogo de ambientes e datasets para RL offline e planejamento em dados estruturados

Versão de 30/09/2026. Cerca de 120 candidatos foram verificados em fontes primárias: paper, código do repositório, página de download e, em vários casos, os próprios dados baixados e medidos. Quando algo não pôde ser verificado, isso está dito na seção 12.

---

## 0. Resumo

- **A pergunta do benchmark.** World models e planejamento superam os baselines offline sem modelo (BC, DQN/FQI, CQL, IQL) quando os dados parecem os de empresas, hospitais e sensores? Tudo é treinado de forma **estritamente offline**.
- **Um ambiente só responde essa pergunta se tiver três coisas:**
  1. dados offline;
  2. uma forma de **avaliar a política sem implantá-la** (simulador, probabilidades das ações registradas nos logs, ou aleatorização);
  3. **mais de um passo de decisão**. Sem horizonte, planejar não traz vantagem.
- **Nenhum ambiente público reúne tudo** (estado tabular heterogêneo, ações da organização, simulador ou probabilidades conhecidas, horizonte longo). O benchmark precisa combinar fontes. Essa combinação faz parte da contribuição.
- **O que existe de mais próximo:**
  - **RL4RS**, a melhor fonte "empresa–cliente": retrato do usuário com 10 colunas categóricas e 32 numéricas, simulador e política de coleta registrada.
  - **NeoRL SalesPromotion e WaterWorks**: modelos aprendidos de dados reais de empresa, com datasets offline prontos em 3 níveis de qualidade.
  - **CityLearn** e **MABIM/ReplenishmentEnv**: sensores e operações, com simulador verdadeiro e estado nomeado de tipos mistos.
  - **Health Gym HIV**: saúde, aberto, 60 passos, tabular de verdade.
  - **Starbucks offers** e **Drink Less**: ações praticamente aleatórias ou com probabilidades registradas.
- **Não jogue fora os 4 ambientes atuais.** Eles viram controles:
  - ICU-Sepsis: derivado de dados reais, com estados discretos;
  - EpiCare: estruturado, mas homogêneo;
  - DTR: físico.
  - O contraste com eles é o que dá força à conclusão.
- **Mudanças que o repositório vai precisar:**
  1. schema com tipo por feature (numérica, categórica, binária, faltante);
  2. **ações contínuas**, porque a maioria dos ambientes de operações e sensores tem ações contínuas;
  3. adaptadores para simuladores aprendidos;
  4. regras de seleção de modelo 100% offline.

---

## 1. A pergunta e o que um ambiente precisa ter

### 1.1 A pergunta
Em dados estruturados do mundo real, com decisões sequenciais e aprendizado apenas a partir de logs:

1. Métodos com modelo (world model + planejamento como CEM/MPC; Dreamer; MOPO/MOReL; e as famílias raw/latent/reconstructive do repo) têm retorno melhor que métodos sem modelo?
2. Esse ganho compensa o custo de treino, de inferência e de chamadas ao planejador?
3. Como o ganho muda com a qualidade dos dados (aleatórios, medianos, especialistas), com o horizonte e com o quão tabular é o estado?

### 1.2 Por que "offline" é inegociável
Hospitais e empresas não podem testar políticas ruins enquanto o algoritmo aprende. Por isso o protocolo precisa garantir três coisas:
- **Treino** usa apenas os logs.
- **Seleção de hiperparâmetros e checkpoints** usa apenas os logs: perda de validação, FQE ou OPE. O simulador não pode ser usado nessa etapa.
- **O simulador ou a OPE** só entram na avaliação final, e isso é declarado.

O repositório já faz quase tudo isso: offline é o modo padrão, a avaliação intermediária no simulador está desligada e o checkpoint é escolhido pela perda de validação. Falta **proibir explicitamente o modo online nos resultados do benchmark**. As recipes online podem continuar existindo, mas como referência.

### 1.3 O que é "dado tabular/estruturado" (checklist)
A base é Grinsztajn et al. 2022, *Why do tree-based models still outperform deep learning on tabular data?*:
- **Colunas heterogêneas**, cada uma com significado próprio. Não vale o mesmo sinal repetido em vários sensores ou pixels.
- **Mistura de tipos**: numéricas, categóricas e binárias, em escalas diferentes, com assimetria e excesso de zeros.
- **Colunas pouco informativas** e relação irregular com o alvo (limiares, degraus).
- **Não invariante a rotação**: misturar colunas destrói informação.
- **Faltantes e ruído de coleta** são comuns.

Os benchmarks tabulares clássicos (Grinsztajn, OpenML-CC18, TabArena) **excluem dados sequenciais**. Aplicar essa noção a trajetórias com decisões é, portanto, uma extensão nova. O precedente mais próximo é o TabReD (2024), com divisões temporais em dados industriais.

Na prática aceitamos dois tipos:
- **"Tabular" (nota 3):** tabelas de empresa, clínica ou cliente.
- **"Estruturado de operação" (nota 2):** variáveis de sensor nomeadas, com calendário categórico, flags binárias e medições em escalas diferentes. É o dado típico de prédios, redes de água, estoques e fábricas.

O que queremos evitar é o "físico puro" (nota 0–1): posições, velocidades e ângulos de robôs, EDOs com poucas variáveis homogêneas, pixels.

---

## 2. Legenda das notas

| Coluna | Significado |
|---|---|
| **T** (tabularidade, 0–3) | 0 = físico/pixels · 1 = poucas variáveis homogêneas · 2 = variáveis nomeadas de tipos mistos, estilo sensor/operação · 3 = tabela heterogênea de empresa/cliente/paciente |
| **E** (avaliabilidade offline da política, 0–3) | 3 = simulador verdadeiro (ou, dentro do benchmark, um simulador fixo) · 2 = logs aleatorizados ou probabilidades registradas (permite OPE) · 1 = só um simulador que você teria de ajustar, ou FQE · 0 = nenhuma |
| **H** (horizonte) | decisões por entidade; 1 = um passo só (bandit) |
| **Acesso** | Aberto · Cadastro (Kaggle, Tianchi, login) · Credenciado (PhysioNet/CITI) · Aplicação (DUA, NDA, carta institucional) |
| **Custo** | esforço de engenharia para integrar: B (baixo) · M (médio) · A (alto) |
| **Papel** | Núcleo · Complementar · Controle · Auxiliar · Descartar |

**Sobre simuladores aprendidos.** Alguns simuladores foram ajustados em dados reais pelos autores: SalesPromotion, WaterWorks, o DIEN do RL4RS, KuaiSim, o simulador do Oralytics. Dentro do benchmark eles funcionam como verdade fixa (E=3). Como retrato do sistema real, porém, são só modelos, e isso precisa ser declarado no paper. A seção 8 explica o viés que isso cria.

---

## 3. O que fazer com os ambientes que você já tem

| Ambiente | Tabular? | Papel recomendado | Por quê | O que fazer |
|---|---|---|---|---|
| **ICU-Sepsis** | Não. O agente vê um índice entre 716 estados; o "vetor de 47" é o centróide do k-means (sexo vira −0,48 ou 0,33) | **Controle, dados reais em forma discreta.** Fica no benchmark | É o único com dinâmica estimada de dados reais de UTI **e** simulador exato, máscara de ações, várias políticas de coleta e política clínica conhecida | Rotular como "MDP tabular derivado do MIMIC-III". Opcional: nomear as 47 colunas do centróide a partir da lista do AI Clinician, sem vender isso como linha real |
| **EpiCare** | Não. 8 sintomas homogêneos em [0,1], com covariância girada por uma **matriz ortogonal aleatória** e arredondados em 0,1 | **Controle estruturado homogêneo.** Fica | Tem observabilidade parcial, efeitos heterogêneos, simulador exato, 4 políticas de coleta com probabilidades registradas e episódios curtos | Nada urgente. Serve para mostrar se o ganho do planejamento depende de o estado ser tabular |
| **DTR AhnChemo** | Não. EDO com 3 floats | **Controle físico.** Fica, mas com prioridade baixa | É exatamente o tipo de estado que o benchmark quer contrastar | **Investigar antes**: o DQN treinado sempre escolhe dose 0 e morre em 100% dos episódios, enquanto a política aleatória dos dados sobrevive em 100% |
| **DTR GhaffariCancer** | Não. EDO com 7 floats em log | **Controle físico.** Prioridade baixa | Idem | Checar se não ficou fácil demais: a política aleatória tem 0,1% de sucesso e o DQN, 100% |

**Resumo:** não tire nenhum. Leve-os para uma trilha **"Controles"** e não os misture na média dos ambientes tabulares. Se precisar economizar GPU, rode os DTR só com o tier `dev`.

---

## 4. Ranking geral, do mais útil ao menos útil

Ordem baseada na utilidade para responder "planejamento contra baselines offline em dados estruturados": primeiro E e H, depois T, acesso e custo.

| # | Ambiente / dataset | Categoria | T | E | H | Acesso | Custo | Papel |
|---|---|---|---|---|---|---|---|---|
| 1 | **RL4RS** (NetEase) | Empresa–cliente | 3 (usuário) / 1 (itens) | 3 (DIEN aprendido) + 2 (política de coleta publicada) | 1–4 páginas × 9 itens | Aberto (CC BY) | M | Núcleo |
| 2 | **NeoRL SalesPromotion** (v1 e v2) | Marketing/cupons | 2 | 3 (aprendido de dados reais) | 30–50 dias | Aberto | B | Núcleo |
| 3 | **CityLearn 3.x** | Energia/prédios | 3 | 3 (verdadeiro) | até 8.760 h | Aberto | M | Núcleo |
| 4 | **MABIM / ReplenishmentEnv** | Estoque/supply chain | 2–3 | 3 (demanda real de 2.778 SKUs) | 100 dias | Aberto | M | Núcleo |
| 5 | **Health Gym HIV** | Saúde crônica | 3 | 1 (você ajusta o simulador) | 60 meses | Aberto (CC BY) | M | Núcleo |
| 6 | **Gerador sintético tabular próprio** | Controle sintético | 3 (controlável) | 3 (verdadeiro) | configurável | Seu | M | Núcleo |
| 7 | **NeoRL WaterWorks** | Sensores/água | 2 | 3 (aprendido de dados reais) | 287 passos de 5 min | Aberto | B | Núcleo |
| 8 | **Starbucks offers** | Marketing/CRM | 3 | 2 (atribuição praticamente uniforme) | 6 ondas | Aberto, licença incerta | B | Núcleo (pequeno) |
| 9 | **Industrial Benchmark** (Siemens) | Indústria | 1–2 | 3 (verdadeiro) + 16 datasets offline | 1.000 | Aberto | B | Complementar |
| 10 | **KuaiRand-Pure + KuaiRec** (+ KuaiSim/DORL) | Empresa–cliente | 2 | 2 (exposições aleatórias, `is_rand`) / 3 (simulador) | sessões longas | Aberto | M–A | Complementar |
| 11 | **Drink Less MRT** | mHealth | 3 | 2 (probabilidades registradas) | 30 dias | Aberto | B | Complementar |
| 12 | **KDD Cup 1998** | Marketing (mala direta) | 3 | 1 (você ajusta o simulador) | 22 campanhas | Aberto | M–A | Complementar |
| 13 | **RetailSynth** | Varejo/promoções | 2 | 3 (verdadeiro) | 50 semanas | Aberto | M | Complementar |
| 14 | **gym-DSSAT** | Agricultura | 2–3 | 3 (verdadeiro) | uma safra (diária) | Aberto | M | Complementar |
| 15 | **Coorte de sepse do MIMIC** (AI Clinician / MIMIC-Sepsis) | UTI | 3 | 1 | ~20 × 4h | Credenciado | M | Complementar (se tiver acesso) |
| 16 | **Insulin4RL** | UTI | 2 | 1 (FQE) | longo, irregular | Credenciado | B–M | Complementar (se tiver acesso) |
| 17 | **Google sbsim** | Prédios (logs reais) | 3 | 3 (simulador calibrado) | ~4.032 | Aberto | A | Complementar (v2) |
| 18 | **AuctionNet** (Alimama) | Anúncios/lances | 2 | 3 (verdadeiro) + dataset de mais de 500M registros | 48 | Aberto | M | Complementar |
| 19 | **SCOPE-RL RTBGym** | Anúncios | 2 | 3 + 2 | 7 | Aberto | B | Teste de sanidade |
| 20 | **STAR\*D / CATIE** (ensaios randomizados em várias etapas, SMART) | Saúde mental | 3 | 2 (randomizado) | 4 etapas / 18 meses | Aplicação (NIMH Data Archive) | A | Complementar (se a UFMG conseguir acesso) |
| 21 | **MetaboNet** (diabetes tipo 1) | Sensores clínicos | 2 | 1 | milhares de passos de 5 min | Cadastro | M | Complementar |
| 22 | **Dunnhumby Complete Journey** | Varejo/cupons | 3 | 1 | 2 anos | Aberto | M | Complementar |
| 23 | **BPI 2017 + SimBank** | Processos (crédito) | 2 | 1 / 3 (SimBank) | por caso | Aberto | M | Complementar |
| 24 | **Grid2Op** | Rede elétrica | 2–3 | 3 | 2.016–8.064 | Aberto | A | Opcional |
| 25 | **Sinergym / BOPTEST** | Prédios | 2 | 3 | ~8.760 | Aberto | M–A | Opcional |
| 26 | **SustainDC** (data center) | Operações | 2 | 3 | 8.760 | Aberto | M | Opcional |
| 27 | **Oralytics** (simulador publicado) | mHealth | 1–2 | 1 | 140 | Aberto | B | Opcional |
| 28 | **Diabetes 130-US** | Hospital | 3 | 1 | 1–2 internações | Aberto | B | Opcional |
| 29 | **Taobao Ad Display (Tianchi 56)** | Anúncios | 2 | 1 | ~23 impressões | Cadastro | A | Opcional |
| 30 | **eICU / HiRID / AmsterdamUMCdb / SICdb / MIMIC-IV-Ext-CLIF** | UTI | 3 | 1 | longo | Credenciado/Aplicação | A | Futuro (v2+) |

As trilhas auxiliares (seções 6.8 e 6.9) e os descartados (6.11) estão fora dessa tabela.

---

## 5. Composição recomendada

### v1: publicável, tudo aberto e integrável em semanas

| Trilha | Ambientes | Como avaliar a política |
|---|---|---|
| **A. Empresa–cliente** | RL4RS · NeoRL SalesPromotion · Starbucks offers | simulador aprendido (DIEN, ensemble do NeoRL) + OPE (RL4RS, Starbucks) |
| **B. Operações e sensores** | CityLearn · MABIM · NeoRL WaterWorks · Industrial Benchmark | simulador verdadeiro ou aprendido |
| **C. Saúde** | Health Gym HIV (semissintético) · Drink Less (OPE) · ICU-Sepsis · EpiCare | simulador ajustado por você / OPE / simulador exato |
| **D. Controle sintético** | gerador tabular próprio · DTR (físico) | simulador verdadeiro |

Na v1 são **14 ambientes** (3 + 4 + 4 + 3, contando os dois DTR), em 4 trilhas, e todos permitem avaliar a política offline.

### v2: amplia com mais custo ou acesso
- KuaiRand + KuaiSim;
- KDD98 com simulador ajustado;
- RetailSynth;
- gym-DSSAT;
- sbsim;
- AuctionNet;
- coorte de sepse do MIMIC e Insulin4RL, se você tiver credencial;
- STAR\*D e CATIE, se a UFMG conseguir o acesso no NIMH Data Archive.

### Trilhas auxiliares (seções 6.8 e 6.9)
- **Bandit de um passo:** Warfarin IWPC, ASSISTments 50 RCTs, Hillstrom, MT-LIFT, Expedia. Aqui o planejamento vira só modelo de recompensa. Serve para separar "o modelo de recompensa ajuda?" de "planejar em vários passos ajuda?".
- **Só fidelidade do world model, sem política:** PhysioNet 2019, AmEx, KKBox, OULAD, Santander. Avalia se o modelo prevê bem trajetórias tabulares reais.

---

## 6. Catálogo por categoria

Cada item traz os dados, o estado, as ações, a forma de avaliar, o acesso e **o que fazer**.

### 6.1 Empresa–cliente: recomendação e e-commerce

#### 1. RL4RS (NetEase, SIGIR 2023 Resource). T 3/1 · E 3+2 · H 1–4 páginas · Aberto · Custo M · **Núcleo**
- **Links:** repo https://github.com/fuxiAIlab/RL4RS · dados https://zenodo.org/record/6622390 (1,98 GB, CC BY 4.0) · paper https://arxiv.org/abs/2110.11073
- **Verificado nos dados (amostra de 20 mil linhas do dataset B):**
  - O `user_protrait` tem 42 dimensões: **10 categóricas** (hash MD5 de valores discretos, cardinalidade de 2 a cerca de 1.100) e **32 numéricas** em log.
  - As numéricas têm muitos zeros (75% a 98,7% em várias colunas) e escalas de 0–1 até 0–2.629. É tabular de verdade, anonimizado como tantos datasets de Kaggle.
- **O que não é tabular:**
  - `item_feature`: 9 itens × 40 floats densos, parecido com embedding;
  - `user_seqfeature`: 67 IDs de itens clicados.
- **Episódios:** Slate (1 página, 9 itens) e SeqSlate (até 4 páginas). Na amostra, média de 2,07 páginas por sessão. Em 91% das sessões o retrato do usuário não muda.
- **Ações:** o sistema escolhe 9 de 283 itens por página, seguindo a regra de "desbloqueio" entre camadas. A recompensa é compra (0/1) × preço.
- **Avaliação:**
  - simulador DIEN treinado nos logs;
  - o **modelo da política de coleta foi publicado**, então dá para reconstruir as probabilidades das ações e fazer OPE;
  - há logs de duas épocas (política supervisionada e política de RL), o que permite estudar mudança de distribuição.
- **Limites:** horizonte curto; a "dinâmica" é basicamente o modelo de resposta do usuário; o simulador é uma rede neural; o stack é TF 1.15 e pede 64 GB de RAM. Você já tem o env conda `rl4rs-sim`.
- **O que fazer:**
  1. **Formular o MDP como 1 item por passo** (9 passos por página, até 36 por sessão). A ação é um item entre 283, com máscara para itens já escolhidos e para a regra de desbloqueio. Isso encaixa no contrato atual (ação discreta + máscara).
  2. **Estado** = retrato do usuário (10 categóricas + 32 numéricas), contexto da página, itens já escolhidos e respostas até ali.
  3. **Converter** os CSVs `@`-separados para o schema do BenchTWM, mantendo `behavior_policy_id` e as probabilidades reconstruídas.
  4. **Avaliar com três evidências:** (a) retorno no simulador DIEN; (b) OPE com as probabilidades da política; (c) fidelidade do world model nos logs de teste.
  5. Rodar o simulador em processo separado (TF 1.15), com um wrapper via subprocess ou socket, para não misturar dependências.

#### 2. KuaiRand-Pure + KuaiRec (Kuaishou). T 2 · E 2/3 · H longo · Aberto · Custo M–A · **Complementar**
- **Links:** https://kuairand.com · https://github.com/chongminggao/KuaiRand · https://github.com/chongminggao/KuaiRec · Zenodo (CC BY-SA)
- **Estado:**
  - 30 colunas de usuário: `user_active_degree`, contagens com versão em faixas, e 18 `onehot_feat` categóricas **criptografadas**, com cardinalidade até 1.471.
  - No KuaiRec, desde janeiro de 2026, há **`user_features_raw.csv`** com colunas nomeadas: gênero, faixa de idade, marca e modelo do celular, preço do aparelho, cidade e nível da cidade, flags de apps instalados.
  - Vídeos: cerca de 50 estatísticas numéricas diárias.
- **Ações:** vídeos expostos pelo sistema. No KuaiRand, **cerca de 0,37% das interações são exposições aleatórias** (`is_rand`), cerca de 1,19 milhão.
- **Avaliação:** a parte aleatória permite estimar recompensas sem viés. Os simuladores KuaiSim, DORL/KuaiEnv e CIRS existem, mas expõem embeddings ao agente.
- **O que fazer:**
  1. Usar o KuaiRand-Pure (194 MB).
  2. O estado passa a ser as colunas cruas do usuário + resumo do histórico recente, e não embeddings.
  3. Avaliar em duas camadas: modelo de resposta calibrado na parte aleatória, e política no KuaiSim ou DORL (declarado como simulador aprendido).
  4. Prioridade v2: o custo de engenharia é alto (ações = milhares de vídeos; é preciso reduzir para um conjunto de candidatos).

#### 3. VirtualTaobao. T 1–2 · E 3 · **Opcional/Descartar**
- **Link:** https://github.com/eyounx/VirtualTaobao
- O cliente são 11 atributos em one-hot (88 binárias) mais 3 dinâmicas, sem nomes. A ação é um vetor de 27 pesos de ranking. O simulador foi aprendido com GAN. O repo está parado desde 2019 e não tem licença.
- **O que fazer:** no máximo um teste de sanidade. Fica fora da v1.

### 6.2 Marketing, promoções, preços e anúncios

#### 4. NeoRL SalesPromotion (Polixir; v1 NeurIPS 2022, v2 2025). T 2 · E 3 · H 30–50 · Aberto · Custo B · **Núcleo**
- **Links:** https://github.com/polixir/NeoRL · https://github.com/polixir/NeoRL2 · dados em `datasets.polixir.site` (funcionando em 30/09/2026) e https://huggingface.co/datasets/polixirai/NeoRL2
- **Estado por usuário:**
  - v1: total de pedidos, média de pedidos, ticket médio, dia da semana (categórico).
  - v2: 8 dimensões.
- **Ações:** número de cupons (0–5) e desconto (0,6–0,95), decididos pela plataforma.
- **Dados:** até 10 mil usuários × 50 dias (500 mil transições, verificado), em 3 níveis de política de coleta (fraca, média, alta).
- **Avaliação:** simulador = ensemble de 5 modelos de usuário aprendidos de dados reais de uma plataforma de delivery.
- **O que fazer:**
  1. Criar um adaptador que carrega `user_model.pt` e expõe `reset`/`step`.
  2. Converter os `.npz` do NeoRL para o schema do BenchTWM.
  3. A ação é contínua em 2 dimensões. Na v1, discretizar (6 quantidades × 8 descontos = 48 ações). Na v2, tratar como contínua.
  4. As probabilidades exatas não são conhecidas ("80% ação média, 20% amostrada"), então OPE não é confiável. Avaliar no simulador.
  5. Cuidado com dependências: `gym` antigo e `attrdict` quebram em Python ≥3.10. Usar o NeoRL2, que usa gymnasium 0.29, ou isolar.

#### 5. Starbucks Rewards offers (Udacity). T 3 · E 2 · H 6 ondas · Aberto (licença incerta) · Custo B · **Núcleo (pequeno)**
- **Link:** dados e notebook em https://github.com/joshuayeung/Starbucks-Capstone-Challenge (espelho do material Udacity)
- **Estado:** `profile` (idade, gênero, renda, data de adesão; 17 mil clientes) + histórico de transações e ofertas.
- **Ações:** 10 ofertas (BOGO, desconto, informativa) ou nenhuma, em 6 ondas. Análise empírica: cerca de 74,6% dos clientes recebem oferta em cada onda, **uniforme entre as 10** e independente de renda e gênero. A probabilidade implícita é ~0,075 por oferta e ~0,25 para nenhuma.
- **Avaliação:** OPE com essas probabilidades inferidas. Os dados são **sintéticos**: o simulador da Udacity não foi publicado.
- **O que fazer:**
  1. Montar o painel cliente × onda com estado = demografia + gasto acumulado + ofertas anteriores.
  2. Testar formalmente a hipótese de atribuição uniforme antes de usar como probabilidade.
  3. Recompensa = gasto líquido na janela da oferta.
  4. Declarar a licença incerta e **não redistribuir**: publicar só o script de download.

#### 6. KDD Cup 1998 (mala direta de doações). T 3 · E 1 · H 22 · Aberto · Custo M–A · **Complementar**
- **Link:** https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html
- **Estado:** 95.412 doadores × 481 colunas: demografia, censo, histórico RFA e códigos nominais. É o mais "de árvore" de todos os candidatos.
- **Ações:** enviar ou não cada uma das 22 campanhas históricas (`ADATE_i`). O resultado é `RAMNT_i`.
- **Precedentes:** Pednault, Abe e Zadrozny (KDD 2002) montaram um MDP com árvores; Jiang e Li (2016) ajustaram um simulador. Nenhum código foi publicado.
- **O que fazer:**
  1. Reconstruir as trajetórias por doador (22 passos).
  2. **Ajustar um simulador semissintético** seguindo a seção 8 (conjunto separado de doadores, famílias de modelo diferentes das avaliadas).
  3. Gerar datasets com políticas de coleta conhecidas: aleatória, "enviar para todos", heurística RFA.

#### 7. RetailSynth. T 2 · E 3 · H 50 semanas · Aberto (Apache-2.0) · Custo M · **Complementar**
- **Links:** https://github.com/RetailMarketingAI/retailsynth · https://arxiv.org/abs/2312.14095
- **Modelo:** visita à loja → categoria → produto → quantidade, calibrado no Dunnhumby, com preços e cupons. A heterogeneidade dos clientes vem de coeficientes latentes, não de demografia.
- **O que fazer:**
  1. Escrever o wrapper gym, que não existe.
  2. Adicionar uma camada demográfica sintética (idade, renda, região) ligada aos coeficientes, para dar tabularidade.
  3. Ação = cupom por cliente, de 0% a 50% em passos de 10%.
  4. Corrigir o uso de `np.int`, que foi removido do NumPy.

#### 8. AuctionNet (Alimama, NeurIPS 2024). T 2 · E 3 · H 48 · Aberto · Custo M · **Complementar**
- **Link:** https://github.com/alimama-tech/AuctionNet
- **Conteúdo:** simulador de leilão e dataset offline com mais de 500M registros (orçamento, restrição de CPA, valor previsto, lance, custo, conversão). Já vem com baselines IQL, BC e Decision Transformer.
- **Ressalva:** o agente é um **anunciante**, não uma empresa agindo sobre clientes.
- **O que fazer:** entra como trilha de "operação com orçamento"; ação contínua (coeficiente de lance).

#### 9. SCOPE-RL RTBGym. T 2 · E 3+2 · H 7 · Aberto · Custo B · **Teste de sanidade**
- **Link:** https://github.com/hakuhodo-technologies/scope-rl
- **Estado:** 7 KPIs nomeados. As políticas de coleta têm **probabilidades conhecidas** e o pacote traz muitos estimadores de OPE.
- **O que fazer:** usar para validar o pipeline de OPE do BenchTWM, não como resultado principal.

#### 10. Dunnhumby The Complete Journey. T 3 · E 1 · H 2 anos · Aberto · Custo M · **Complementar**
- **Link:** https://www.dunnhumby.com/source-files/
- **Conteúdo:** 2.500 domicílios, demografia em faixas, campanhas e cupons enviados pela empresa, direcionados pelo histórico de compras (portanto confundidos). Não há probabilidades.
- **O que fazer:** painel domicílio × semana. Serve para a trilha de fidelidade do world model, ou como base para ajustar um simulador (é a base do RetailSynth).

#### 11. Taobao Ad Display/Click (Tianchi 56). T 2 · E 1 · H ~23 · Cadastro · Custo A · **Opcional**
- **Conteúdo:** perfil de usuário todo categórico (segmento, gênero, idade, nível de consumo, ocupação), anúncios exibidos em 8 dias e 22 dias de comportamento.
- **Limites:** sem aleatorização e com ~840 mil anúncios possíveis como ação.
- **O que fazer:** só se sobrar tempo, reduzindo as ações a categorias de anúncio.

### 6.3 Operações, indústria, sensores e energia

#### 12. CityLearn 3.x. T 3 · E 3 · H até 8.760 · Aberto (MIT) · Custo M · **Núcleo**
- **Links:** https://github.com/intelligent-environments-lab/CityLearn (ativo, commit em 22/09/2026) · https://www.citylearn.net
- **Estado:** cerca de 290 sinais nomeados.
  - Categóricos: mês, tipo de dia, hora, modo do HVAC.
  - Binários: horário de verão, queda de energia, veículo elétrico conectado.
  - Contínuos: temperaturas, estado de carga, preço, intensidade de carbono, carga, solar, previsões.
- **Ações:** baterias, carregadores de veículos elétricos, aquecimento/resfriamento e cargas adiáveis (contínuas).
- **O que fazer:**
  1. Gerar datasets offline com os controladores de regra incluídos, mais versões com ruído, em 3 níveis de qualidade.
  2. Fixar a versão, porque a API muda entre versões.
  3. Ações contínuas: discretizar por dispositivo na v1, suporte contínuo na v2.
  4. Avaliar com os KPIs do próprio CityLearn e com o retorno.

#### 13. MABIM / ReplenishmentEnv (Microsoft Research Asia). T 2–3 · E 3 · H 100 · Aberto · Custo M · **Núcleo**
- **Links:** https://github.com/VictorYXL/ReplenishmentEnv · https://arxiv.org/abs/2306.07542
- **Estado por SKU:** preço de venda, custo de compra, lead time, volume, custo por pedido, estoque, em trânsito, excesso, e 21 dias de demanda. A **demanda é real** (2.778 SKUs, 2018–2021).
- **Ações:** quantidade a pedir como múltiplo da demanda média. Há cenários de 50 a 2.000 SKUs e até 3 lojas.
- **O que fazer:**
  1. Gerar dados offline com as políticas base-stock e (s,S) incluídas, mais ruído.
  2. Começar com 1 loja e um subconjunto de SKUs, com um agente por SKU ou compartilhado.
  3. Discretizar a quantidade em bins (0, 0,5, 1, 1,5, 2 ... × demanda média).
  4. Respeitar as divisões temporais do repo (treino até jun/2021).
- **Atenção:** o repo não é mantido desde 2023. Congele as dependências.

#### 14. NeoRL WaterWorks. T 2 · E 3 · H 287 · Aberto · Custo B · **Núcleo**
- **Estado:** 14 dimensões com vazões e pressões de 5 estações + temperatura, dia da semana, feriado e tempo. É dado de sensor de uma rede real de água.
- **Ações:** 4 pressões de bomba (contínuas).
- **Avaliação:** modelo ONNX aprendido de dados reais. Datasets em 4 níveis (baixo, médio, alto, especialista).
- **O que fazer:** mesmo adaptador do SalesPromotion; discretizar ou suportar ações contínuas.

#### 15. Industrial Benchmark (Siemens). T 1–2 · E 3 · H 1.000 · Aberto · Custo B · **Complementar**
- **Links:** https://github.com/siemens/industrialbenchmark (branch `offline_datasets`) · https://arxiv.org/abs/1709.09480
- **Estado:** setpoint, velocity, gain, shift, fatigue, consumption. São nomeados, mas todos contínuos na faixa 0–100.
- **Ações:** 3 deltas contínuos.
- **Dados:** 16 datasets offline (políticas ruim, mediana e otimizada, com 0% a 80% de ações aleatórias, mais uma 100% aleatória).
- **O que fazer:** é a âncora "industrial mas homogênea". Mostra o comportamento em sensores sem heterogeneidade de tipos.

#### 16. gym-DSSAT (agricultura). T 2–3 · E 3 · H uma safra · Aberto · Custo M · **Complementar**
- **Link:** https://gitlab.inria.fr/rgautron/gym_dssat_pdi
- **Estado:** 17 variáveis: estágio de crescimento (categórico), dias após o plantio, biomassa, índice de área foliar, estresse hídrico e de nitrogênio, fertilizante acumulado, clima, solo.
- **Ações:** nitrogênio e irrigação diários.
- **O que fazer:** gerar dados com políticas agronômicas de regra mais ruído. Roda via Docker.

#### 17. Google sbsim (Smart Buildings). T 3 · E 3 · H ~4.032 · Aberto · Custo A · **v2**
- **Links:** https://github.com/google/sbsim · https://www.tensorflow.org/datasets/catalog/smart_buildings
- **Diferencial:** é o **único com logs reais** de automação predial (sensores, setpoints e recompensa, 2019–2024) **e** um simulador calibrado.
- **O que fazer:** v2. Os dados são grandes (11 GiB baixados, 87 GiB preparados) e dependem de TF-Agents.

#### 18. Opcionais de operação
- **Grid2Op:** estado de tipos mistos (calendário, status de linha binário, topologia inteira, fluxos) e simulador verdadeiro. O espaço de ações é enorme.
- **Sinergym e BOPTEST:** EnergyPlus e Modelica; lentos.
- **SustainDC:** data center.
- **gym-invmgmt:** estoque.
- **mobile-env:** telecom.
- **O que fazer:** só depois da v1.

### 6.4 Saúde: UTI e hospital

#### 19. Coorte de sepse do MIMIC (AI Clinician / microsoft/mimic_sepsis / MIMIC-Sepsis 2025). T 3 · E 1 · H ~20 × 4h · Credenciado · Custo M · **Complementar**
- **Links:**
  - https://github.com/matthieukomorowski/AI_Clinician
  - https://github.com/microsoft/mimic_sepsis
  - https://github.com/yongh7/MIMIC-sepsis (MIMIC-IV v3.1, 35.239 pacientes)
  - https://github.com/cmudig/AI-Clinician-MIMICIV
- **Estado:** 47 colunas nomeadas: 4 binárias, 32 contínuas padronizadas (vitais, laboratório, SOFA, SIRS) e 11 em log.
- **Ações:** fluido IV × vasopressor em grade 5×5. **Recompensa:** mortalidade em 90 dias.
- **Avaliação:** não há simulador e a política dos médicos é desconhecida (precisa ser estimada). Usar FQE, OPE com probabilidades estimadas e simulador semissintético.
- **O que fazer:** se você tiver credencial no PhysioNet, é a versão "linha real" do ICU-Sepsis. Vi que já existe código do AI_Clinician em `/scratch/pedrobacelar/bracis-sepse-wm`.

#### 20. Insulin4RL (2026). T 2 · E 1 (FQE) · H longo e irregular · Credenciado · Custo B–M · **Complementar**
- **Links:** https://physionet.org/content/insulin4rl/1.0.0/ · https://github.com/tdgfrost/insulin4rl
- **Conteúdo:** 12.209 pacientes do MIMIC-IV, 376 mil decisões de insulina, 140 features, ações de ajuste de dose (manter, parar, ±0,5 a ±5 U/h).
- **Avaliação:** já traz baselines BC, IQL e CQL avaliados com FQE.
- **O que fazer:** entra pronto se você tiver credencial. Acrescenta o caso de tempo irregular.

#### 21. Health Gym: sepse e hipotensão. T 3 · E 1 · Credenciado · **Opcional**
- Versões sintéticas (GAN) do MIMIC-III. Não têm simulador nem recompensa. Na hipotensão, 97% das ações de fluido caem no bin zero.
- **O que fazer:** priorize a versão HIV (aberta). Estas duas só se você tiver credencial.

#### 22. EHRs grandes: eICU, HiRID, AmsterdamUMCdb, SICdb, MIMIC-IV-Ext-CLIF, MIMIC-BR. T 3 · E 1 · **Futuro**
- **Diferenciais:** o **MIMIC-IV-Ext-CLIF** (v1.1.0, março de 2026) tem 14 tabelas Parquet padronizadas e 89 categorias de medicação, uma boa base para espaços de ação. O eICU (208 hospitais) serve para testar generalização entre hospitais.
- **Ressalva:** o **MIMIC-BR** (v1.0.1) tem só 87 óbitos em cerca de 29 mil pacientes e ~48 medições por paciente. Suspeito de estar incompleto.
- **O que fazer:** v2+. Exigem pipeline próprio de extração e não têm como avaliar a política diretamente.

### 6.5 Saúde: doenças crônicas, mHealth e ensaios randomizados

#### 23. Health Gym HIV. T 3 · E 1 · H 60 · Aberto (CC BY 4.0) · Custo M · **Núcleo**
- **Links:** https://doi.org/10.6084/m9.figshare.22827878 · paper https://arxiv.org/abs/2203.06369
- **Verificado nos dados:** 8.916 pacientes × 60 meses (534.960 linhas).
  - Numéricas com cauda muito pesada: carga viral (mediana 39, máximo ~97 mil) e CD4.
  - Binárias: gênero e flags de "medido" (13% a 15% medidos).
  - Categóricas: etnia e os 5 componentes do regime de drogas.
- **Ações:** o regime de drogas. São **135 combinações**; as 25 mais comuns cobrem 92,7% dos passos. A mediana é de 5 trocas de regime por paciente.
- **Limites:** dados gerados por GAN (a dinâmica não é causal), sem recompensa e sem simulador.
- **O que fazer:**
  1. Ação discreta = as 25 combinações mais comuns + "outra", com máscara.
  2. Recompensa definida por você: supressão viral (carga < 50) e ganho de CD4, penalizando trocas.
  3. Ajustar um **simulador semissintético** (seção 8) em uma parte separada dos pacientes.
  4. Gerar datasets com 3 políticas de coleta de probabilidades conhecidas (clonagem do comportamento, epsilon-clínica, aleatória), exatamente como o repo já faz para o ICU.

#### 24. Drink Less MRT (UCL). T 3 · E 2 · H 30 · Aberto · Custo B · **Complementar**
- **Links:** https://osf.io/w3szp/ · https://github.com/lauren-bell/DrinkLessMRT
- **Conteúdo:** 349 pessoas × 30 dias, um ensaio micro-randomizado.
  - Covariáveis: idade, gênero, tipo de emprego, escore AUDIT, e variáveis diárias (dias desde a instalação, uso ontem, notificação ontem).
- **Ações:** notificação diária às 20h (nenhuma, padrão ou nova), com **probabilidades registradas** (0,6 e 0,3).
- **Limite:** a ação pouco altera o estado futuro, então o problema fica perto de um bandit.
- **O que fazer:** OPE exata com as probabilidades do ensaio. Arquivos `.rds` (ler com pyreadr em UTF-8).

#### 25. STAR\*D (depressão) e CATIE (esquizofrenia). T 3 · E 2 · H 4 etapas / 18 meses · Aplicação (NIMH Data Archive) · Custo A · **Se conseguir acesso**
- São os **únicos dados clínicos reais randomizados em várias etapas** (SMART). O STAR\*D tem 4.041 pacientes, até 4 níveis de tratamento e escalas repetidas (QIDS, HRSD).
- **O que fazer:** pedir acesso no NIMH Data Archive via certificação institucional (desde 2025 exige atestar segurança NIST 800-171 ou ISO 27001). Se sair, vira um ponto forte do paper.

#### 26. MetaboNet (diabetes tipo 1, 2026). T 2 · E 1 · H milhares de passos · Cadastro · Custo M · **Complementar**
- **Links:** https://arxiv.org/abs/2601.11505 · https://metabo-net.org
- **Conteúdo:** 3.135 pacientes (1.895 públicos). Linhas a cada 5 minutos com glicose contínua (CGM), insulina basal e bolus, carboidratos, exercício, frequência cardíaca e demografia.
- **O que fazer:** trilha "sensores clínicos". Ação contínua (insulina) e avaliação por FQE ou simulador aprendido. Para ter a dinâmica verdadeira, usar junto o Simglucose do NeoRL-2 (T 1, E 3).

#### 27. Opcionais de saúde
- **Oralytics:** o simulador ajustado no ensaio foi publicado (E 1); o piloto tem só 9 usuários.
- **HeartSteps V1:** 37 usuários, probabilidades conhecidas (E 2). Pequeno.
- **Diabetes 130-US:** tabular (T 3), mas com 1–2 internações por paciente.
- **Sepsis Cases:** event log tabular com 1.050 casos. Pequeno.
- **Oberst sepsis:** 8 variáveis categóricas, simulador exato e política de coleta conhecida (E 3). É um brinquedo, útil como controle sintético.
- **MedGym-TaCoS:** simulador PINN, 6 variáveis contínuas.

### 6.6 Processos de negócio e finanças

#### 28. BPI Challenge 2017 + SimBank. T 2 · E 1 / 3 · Aberto · Custo M · **Complementar**
- **Links:** https://data.4tu.nl/articles/_/12696884/1 · https://github.com/JakobDeMoorKULstudent/SimBank · ProCause https://github.com/JakobDeMoorKULstudent/ProCause
- **BPI 2017:** 31.509 pedidos de empréstimo. O banco faz ofertas com valor, prazo e parcela, e o resultado é a aceitação.
- **SimBank:** simulador do mesmo tipo de processo, com **contrafactual verdadeiro** e confusão ajustável. Tem poucas colunas.
- **O que fazer:** trilha de "processos". SimBank para avaliar, BPI 2017 para a fidelidade do world model em logs reais.

#### 29. Finanças com painéis ao longo do tempo (Freddie Mac, Home Credit, Lending Club, Berka). T 3 · E 0–1
- Tabulares de verdade, mas com ações observacionais (renegociação, limite de crédito) ou de um passo (aprovar ou recusar).
- **O que fazer:** trilha de fidelidade do world model (6.9), ou bandit (Lending Club). Não entram na trilha de política.

### 6.7 Controle sintético: gerador tabular próprio (recomendado)

**Por quê:** é a única forma de ter ao mesmo tempo (1) estado tabular heterogêneo, (2) dinâmica **verdadeira conhecida**, (3) horizonte longo e (4) controle sobre o quão tabular o problema é. Também elimina o viés de simulador aprendido (seção 8).

**Como desenhar:**
- **Entidade** (cliente ou paciente) com:
  - atributos estáticos **categóricos** (segmento, região, plano);
  - variáveis dinâmicas **numéricas** com excesso de zeros e cauda pesada (gasto, saldo, uso);
  - **binárias** (inadimplente, ativo);
  - **colunas inúteis** e **faltantes**.
- **Dinâmica** com **limiares e interações** (por exemplo, uma oferta só funciona se o segmento for X e o saldo passar de Y), efeitos atrasados e um **estado latente oculto** (observabilidade parcial).
- **Ações da organização** (ofertas, contato, preço), discretas, com máscara quando fizer sentido.
- **Políticas de coleta** com probabilidades exatas: aleatória, heurística, quase ótima, e misturas.
- **Botões de dificuldade:**
  - rotação das features, de 0 (tabular) a 1 (homogêneo, estilo EpiCare);
  - fração de colunas inúteis;
  - ruído;
  - horizonte;
  - suporte dos dados (quão diversa é a política de coleta).
- **Resultado:** curvas de "ganho do planejamento × tabularidade × qualidade dos dados", que é um resultado central e controlado.

### 6.8 Trilha auxiliar: bandits de um passo (H = 1)
Servem para separar "o modelo de recompensa ajuda?" de "planejar ajuda?".

| Dataset | T | E | Observação |
|---|---|---|---|
| **Warfarin IWPC** | 3 | 3 (dose correta conhecida) | 5.700 pacientes × 68 colunas; aberto via ClinPGx |
| **ASSISTments 50 RCTs** | 3 | 2 | cerca de 30 mil alunos; não pode ser redistribuído |
| **Hillstrom MineThatData** | 3 | 2 | 64 mil clientes, e-mail aleatorizado em 3 braços |
| **MT-LIFT (Meituan)** | 1–2 | 2 | 5,5M linhas, cupons por RCT, features anonimizadas |
| **Expedia Hotel Searches** | 3 | 2 (parte com ordenação aleatória) | slate; viés para buscas com compra |
| **Open Bandit (ZOZOTOWN)** | 1 | 2 | features hasheadas, sem ID de usuário |
| **Duolingo notificações** | 1 | 2 (uniforme) | 202M linhas, features fracas |

### 6.9 Trilha auxiliar: fidelidade do world model sem ações
Mede se o modelo prevê bem trajetórias tabulares reais: erro em um passo e erro no rollout, por tipo de feature.

- **PhysioNet 2019:** aberto, 40 variáveis horárias, 40 mil pacientes, sem tratamento.
- **AmEx Default:** ~190 features, 11 categóricas, extratos mensais.
- **KKBox:** assinaturas e uso diário.
- **OULAD:** alunos com demografia e cliques semanais.
- **Santander Product:** 17 meses × 24 atributos.
- **Freddie Mac:** painel mensal de empréstimos.

### 6.10 Controles não tabulares (os atuais)
ICU-Sepsis, EpiCare, DTR AhnChemo e DTR GhaffariCancer. Detalhes na seção 3. Opcionalmente, somar o Oberst sepsis e o NeoRL-2 Simglucose.

### 6.11 Descartados, e por quê

| Descartado | Motivo |
|---|---|
| Spotify MSSD | não está mais disponível para download |
| RecSim / RecSim NG / RecoGym / AuctionGym / Deezer | estado são vetores latentes sintéticos |
| Yahoo R6 | features latentes; a página não resolve mais |
| Criteo Uplift / Avazu / Criteo CTR | features projetadas ou anonimizadas, um passo, sem ação |
| EdNet / Junyi / Tmall / UserBehavior / JData | só IDs ou só ações do usuário |
| Coupon Purchase (Ponpare) | não registra o que o sistema mostrou a cada usuário |
| Online Retail I/II | sem ações; só IDs e quantidade |
| DiDi KDD Cup 2020 | o simulador nunca foi publicado |
| FinRL / ABIDES | séries de preço (não tabular) |
| pc-gym / Tennessee Eastman / RWRL / NeoRL-2 físicos / sumo-rl / CityFlow | estado físico ou homogêneo |
| EduGym | é um ambiente para ensinar RL, não de educação |
| ADHD SMART (DTRlearn2) | dados simulados |
| PaySim / AMLSim | só ações do usuário; poucas colunas |

---

## 7. Protocolo offline

1. **Dados:** divisões por entidade, sem sobreposição entre treino, validação e teste de OPE. Nos dados reais, usar divisão temporal quando fizer sentido (recomendação do TabReD).
2. **Qualidade dos dados:** pelo menos 3 níveis de política de coleta por ambiente (fraca, mediana, boa), mais uma mistura. O NeoRL já traz isso; nos outros, você gera.
3. **Seleção de modelo:** só com critérios offline (perda de validação, FQE, OPE). **É proibido** usar o simulador para escolher hiperparâmetros ou checkpoints. Para cada método, reportar também o resultado com um orçamento fixo de hiperparâmetros, sem ajuste.
4. **Avaliação final:** simulador (quando houver) e OPE (quando houver probabilidades), como **evidências separadas**.
5. **Baselines sem modelo:** BC, DQN/FQI, CQL, IQL e, com ações contínuas, TD3+BC.
6. **Métodos com modelo:** as famílias raw, latent e reconstructive com CEM, o Dreamer e, com ações contínuas, MOPO, MOReL e COMBO.
7. **Custo:** registrar tempo de treino, tempo de inferência e chamadas ao planejador (o repo já faz).
8. **Nunca tirar média entre ambientes de famílias diferentes**; comparar dentro de cada ambiente e agregar por ranking (como o repo já define).

---

## 8. Como avaliar sem implantar

| Camada | Onde se aplica | O que mede | Risco |
|---|---|---|---|
| **(a) Simulador verdadeiro** | CityLearn, MABIM, IB, gym-DSSAT, RetailSynth, AuctionNet, gerador próprio, ICU-Sepsis, EpiCare, DTR | retorno real da política | a distância entre simulador e mundo real |
| **(b) Simulador aprendido** | RL4RS (DIEN), NeoRL SalesPromotion e WaterWorks, KuaiSim, HIV e KDD98 (ajustados por você) | retorno "semissintético" | **viés a favor de métodos com modelo** (ver abaixo) |
| **(c) OPE com probabilidades registradas** | RL4RS, Starbucks, Drink Less, RTBGym, STAR\*D, bandits | retorno estimado nos dados reais | variância alta; exige cobertura das ações pela política de coleta |
| **(d) Fidelidade do world model** | todos os logs reais | erro em um passo e no rollout, por tipo de feature | não mede a política |

**A armadilha do simulador aprendido.** Se o simulador foi ajustado nos mesmos dados que o world model vê, o world model pode simplesmente reaprender o simulador. O planejador então explora os erros de extrapolação do próprio simulador, o que infla os métodos com modelo. Como mitigar:
- ajustar o simulador em uma **parte separada de entidades**;
- usar **famílias de modelo diferentes** das avaliadas (ensemble misto);
- **injetar estocasticidade calibrada**;
- validar o simulador em estatísticas de trajetória dos dados reais;
- reportar essa trilha separadamente como "semissintética".

---

## 9. Mudanças necessárias no repositório

| # | Mudança | Por quê | Onde |
|---|---|---|---|
| 1 | **Schema v3 com features tipadas** (tipo, cardinalidade, máscara de faltantes, arrays separados para numéricas e categóricas) | hoje só existe `float32 (N, d)` | `src/offline/dataset.py`, `generate_dataset.py`, `docs/data-schema.md` |
| 2 | **`EnvSpec.feature_schema`** | cada adaptador declara os tipos | `src/envs/base.py` |
| 3 | **Encoder tabular compartilhado** (embedding de categóricas, normalização por quantis, flag de faltante) | as redes atuais recebem o vetor cru, sem normalizar | `src/models/` |
| 4 | **Ações contínuas** | IB, WaterWorks, CityLearn, MABIM, SalesPromotion, gym-DSSAT e AuctionNet têm ações contínuas | schema, trainers, planejador CEM (versão gaussiana), TD3+BC |
| 5 | **Adaptador genérico para simulador aprendido** (carregar um modelo, reset a partir de estados iniciais dos dados) | NeoRL, DIEN, HIV, KDD98 | `src/envs/` |
| 6 | **Importador de datasets externos** (NeoRL `.npz`, RL4RS CSV, IB pickles) | nem todo dado é gerado pelo repo | `src/offline/` |
| 7 | **OPE com probabilidades externas** (ler colunas de propensão dos logs) | RL4RS, Starbucks, Drink Less | `src/offline/ope.py` |
| 8 | **Métricas de fidelidade por tipo** (RMSE para numéricas, log-loss/acurácia para categóricas, calibração para binárias) | camada (d) | `src/evaluation.py` |
| 9 | **Travas do protocolo offline** (bloquear avaliação no simulador durante a seleção; marcar resultados online como "referência") | seção 7 | `src/operations/` |
| 10 | Corrigir o comportamento do **DQN no AhnChemo** | é pior que a política que gerou os dados | investigar escala das features e recompensa |

**Ordem sugerida:**
1. (1–3), com o **gerador sintético próprio** como primeiro ambiente novo: é o mais barato e valida o schema.
2. Health Gym HIV e Starbucks (tabulares, ação discreta).
3. RL4RS.
4. Suporte a ações contínuas.
5. NeoRL (SalesPromotion, WaterWorks, IB), CityLearn e MABIM.

---

## 10. Faz sentido? Posicionamento

**Sim.** O gap é defensável: não achei nenhum benchmark que compare **planejamento e world models contra RL offline sem modelo** em **estados tabulares heterogêneos**, com protocolo 100% offline. Os vizinhos que precisam ser citados e diferenciados:

| Trabalho | O que faz | Diferença para o BenchTWM |
|---|---|---|
| **NeoRL / NeoRL-2** (2022, 2025) | RL offline "quase real" | não foca em dados tabulares nem em planejamento contra métodos sem modelo; poucos domínios de empresa |
| **medDreamer** (2025) | Dreamer em EHR irregular | um único método, sem benchmark, sem código público |
| **EHRWorld** (2026) | world model com LLM em eventos | sequência de eventos, não linhas tabulares |
| **RL4RS / KuaiSim** | RL em recomendação | um domínio só |
| **Insulin4RL** (2026) | RL offline em insulina | sem trilha de world model |
| **GBRL** (NVIDIA, 2025) | árvores em RL | ambientes de jogo, online |
| **TabReD** (2024) | tabular com divisões temporais | sem ações |
| **CausalLongPFN** (2026) | contrafactuais longitudinais | inferência causal, não RL |

**Terminologia:** na literatura de RL, "tabular" quer dizer "espaço de estados finito". Use "**structured/tabular-feature states**" no paper para evitar confusão.

---

## 11. Correções à conversa antiga

| A conversa antiga dizia | O que foi verificado |
|---|---|
| ICU-Sepsis "A" | serve para RL, mas **não é tabular** (índice discreto) |
| EpiCare "A forte" | serve para RL, mas **não é tabular** (homogêneo, rotacionado) |
| DTR-Bench "observação estruturada" | são EDOs com 1 a 7 floats; o ambiente MIMIC não existe |
| Health Gym "A/B" | é **B**: GAN, sem simulador e sem recompensa; só o HIV é aberto |
| RL4RS "A forte" | confirmado. O retrato do usuário é tabular (verificado nos dados), os itens não |
| KuaiRand "A forte" | é A−/B+: só 0,37% de exposições aleatórias e features criptografadas |
| Spotify MSSD | **não está mais disponível** |
| RecSim NG / RecoGym "A forte" | não são tabulares; o RecSim NG foi arquivado |
| EdNet "A/B forte" | não é tabular (só IDs) |
| Open Bandit "A" | é de um passo e não tem ID de usuário |
| MIMIC-IV-Ext-CLIF / MIMIC-BR | existem; o MIMIC-BR parece incompleto |
| Berka / Criteo | links oficiais fora do ar; há espelhos |
| Ficaram de fora | NeoRL, CityLearn, MABIM, Starbucks, Drink Less, KDD98, RetailSynth, AuctionNet, Insulin4RL, MetaboNet, STAR\*D/CATIE, Warfarin, gerador próprio |

---

## 12. O que não foi verificado

- Páginas de competição do Kaggle (AmEx, KKBox, Expedia, Ponpare, Avazu): a API exige login. Esquemas vieram de espelhos e do Wayback.
- Licenças: Starbucks, MT-LIFT, Berka, Eedi, Dunnhumby.
- Tianchi (Taobao 56, O2O): a página é renderizada em JavaScript.
- A lista exata de colunas dos arquivos credenciados do Health Gym (sepse e hipotensão).
- Comprimento exato dos episódios do gym-DSSAT, dos datasets do CityLearn e do Grid2Op.
- O fluxo de download do MetaboNet; o link do dataset AuctionNet; o conteúdo do Intern Health Study no openICPSR.
- Se os pacotes do NeoRL, MABIM e RetailSynth instalam no Python 3.13. Só foram verificados downloads e código.
- O motivo de o MIMIC-BR ter tão poucos óbitos.

---

## 13. Fontes principais (seleção)

- Grinsztajn et al. 2022: https://arxiv.org/abs/2207.08815 · TabReD: https://arxiv.org/abs/2406.19380
- RL4RS: https://arxiv.org/abs/2110.11073 · https://github.com/fuxiAIlab/RL4RS
- NeoRL: https://github.com/polixir/NeoRL · NeoRL-2: https://arxiv.org/abs/2503.19267
- CityLearn: https://github.com/intelligent-environments-lab/CityLearn
- MABIM: https://arxiv.org/abs/2306.07542 · https://github.com/VictorYXL/ReplenishmentEnv
- Industrial Benchmark: https://arxiv.org/abs/1709.09480 · https://github.com/siemens/industrialbenchmark
- Health Gym: https://arxiv.org/abs/2203.06369 · HIV: https://doi.org/10.6084/m9.figshare.22827878
- KuaiRand/KuaiRec: https://kuairand.com · https://github.com/chongminggao/KuaiRec
- KDD Cup 98: https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html
- RetailSynth: https://arxiv.org/abs/2312.14095 · AuctionNet: https://github.com/alimama-tech/AuctionNet
- SCOPE-RL: https://github.com/hakuhodo-technologies/scope-rl
- Drink Less: https://osf.io/w3szp/ · Insulin4RL: https://physionet.org/content/insulin4rl/1.0.0/
- MetaboNet: https://arxiv.org/abs/2601.11505 · MIMIC-Sepsis: https://arxiv.org/abs/2510.24500
- MIMIC-IV-Ext-CLIF: https://physionet.org/content/mimic-iv-ext-clif/1.1.0/
- gym-DSSAT: https://gitlab.inria.fr/rgautron/gym_dssat_pdi · sbsim: https://github.com/google/sbsim
- SimBank: https://arxiv.org/abs/2506.14772 · BPI 2017: https://data.4tu.nl/articles/_/12696884/1
- medDreamer: https://arxiv.org/abs/2505.19785 · GBRL: https://arxiv.org/abs/2407.08250
