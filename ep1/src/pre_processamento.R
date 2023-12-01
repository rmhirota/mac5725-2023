
# Leitura dos dados
df <- "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv" |>
  readr::read_csv(col_select = c("review_text", "overall_rating")) |>
  dplyr::rename(texto = review_text, rotulo = overall_rating)

# Filtrar dados e tratamento básico
df <- df |>
  dplyr::filter(dplyr::between(rotulo, 0, 5), !is.na(texto)) |>
  dplyr::mutate(texto = stringr::str_squish(tolower(texto)))

# Partilha

set.seed(923)
df_split <- rsample::initial_validation_split(df, c(.65, .10))
df_split
readr::write_csv(rsample::training(df_split), "ep1/src/base_treino.csv")
readr::write_csv(rsample::validation(df_split), "ep1/src/base_validacao.csv")
readr::write_csv(rsample::testing(df_split), "ep1/src/base_teste.csv")


