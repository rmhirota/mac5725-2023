# EP 1: RNNs Bidirecionais, Overfitting, Underfitting

## Pré-processamento

Os dados utilizados fazem parte do corpus de avalia̧cões da B2W, disponibilizados no [Github](https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv).

Para realizar o pré-processamento dos dados, basta executar o código em R em `src/1_pre_processamento.R`. 

## Treino

O treino dos modelos é feito no script disponível em `src/2_treino.py`.

## Teste

O teste com os modelos selecionados são rodados no script `src/3_modelos_finais.py`.

## Relatório

O relatório foi feito em [Quarto](https://quarto.org/) (`relatorio-ep1.qmd` gera o arquivo em PDF correspondente `relatorio-ep1.pdf`), basta executar o comando `quarto render relatorio-ep1.qmd` no diretório raiz.