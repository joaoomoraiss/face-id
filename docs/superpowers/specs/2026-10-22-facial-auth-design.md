# FaceCheck — autenticação facial com FastAPI

Data: 2026-10-22

API HTTP para cadastro e login de usuários por reconhecimento facial. O rosto vira um embedding na hora do cadastro. O login compara a foto enviada com o embedding daquele e-mail. Não há senha.

## Decisões fechadas

- Login bem-sucedido devolve um JWT. As rotas de usuário exigem `Authorization: Bearer`.
- O cadastro público e a criação feita por admin usam o mesmo fluxo: nome, e-mail e uma foto.
- Todo cadastro nasce com `role` igual a `user`. A promoção a `admin` é feita direto no MongoDB. A API não altera `role`.
- `user` lê e atualiza só o próprio perfil.
- `admin` lista, lê, cria, atualiza e apaga qualquer usuário.
- Login é verificação 1:1: e-mail mais foto. A comparação usa apenas o embedding daquele usuário.
- Credenciais e segredos ficam no `.env`. O código sobe e fala com o MongoDB assim que a URI estiver preenchida.

## Fora de escopo

- Frontend.
- Senha, refresh token e recuperação de conta.
- Endpoint para promover admin.
- Identificação 1:N (foto sem e-mail).
- Armazenar o arquivo da foto. Só o embedding fica no banco.

## Stack

- FastAPI e Uvicorn.
- MongoDB via Motor e Beanie.
- DeepFace para extrair o embedding.
- PyJWT para o access token.
- Pydantic Settings para o `.env`.
- `python-multipart` para upload.

## Layout

```text
app/
  main.py
  core/config.py
  core/security.py
  db/mongodb.py
  models/user.py
  schemas/user.py
  schemas/auth.py
  api/deps.py
  api/v1/router.py
  api/v1/endpoints/auth.py
  api/v1/endpoints/users.py
  services/face.py
  services/users.py
tests/
  conftest.py
  test_face.py
  test_auth.py
  test_users.py
.env.example
.gitignore
requirements.txt
README.md
```

`app/main.py` cria a aplicação, registra o router em `/api/v1` e abre o MongoDB no lifespan. Se a URI existe e o ping inicial falha, o processo não termina de subir. `GET /health`, na raiz e fora de `/api/v1`, responde `{"status": "ok"}` quando o ping passa e `503` quando o ping falha depois da subida.

## Configuração

O processo lê o `.env` na raiz. Se `MONGODB_URI` ou `JWT_SECRET_KEY` estiver vazio, a aplicação recusa subir e indica a variável ausente.

| Variável | Papel | Exemplo no `.env.example` |
| --- | --- | --- |
| `MONGODB_URI` | Conexão | vazio, preenchido depois |
| `MONGODB_DB_NAME` | Database | `facecheck` |
| `JWT_SECRET_KEY` | Assinatura do token | vazio, preenchido depois |
| `JWT_ALGORITHM` | Algoritmo | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Validade | `60` |
| `FACE_MODEL_NAME` | Modelo DeepFace | `Facenet512` |
| `FACE_DETECTOR_BACKEND` | Detector | `opencv` |
| `FACE_DISTANCE_METRIC` | Métrica | `cosine` |
| `FACE_DISTANCE_THRESHOLD` | Corte da distância | `0.30` |

`.env` entra no `.gitignore`. `.env.example` vai para o repositório, sem segredo.

`FACE_MODEL_NAME` precisa permanecer o mesmo depois que existirem usuários. Embeddings de modelos diferentes não são comparáveis. A métrica implementada é só `cosine`. Qualquer outro valor de `FACE_DISTANCE_METRIC` impede a subida. O limiar `0.30` é o corte de cosseno do Facenet512 usado pelo DeepFace: distância menor ou igual ao limiar significa a mesma pessoa.

## Documento `users`

| Campo | Tipo | Regra |
| --- | --- | --- |
| `name` | string | Obrigatório, sem ficar só com espaços, no máximo 120 caracteres |
| `email` | string | Obrigatório, formato de e-mail, único, guardado em minúsculas |
| `role` | string | `user` ou `admin`. Cadastro grava `user` |
| `embedding` | lista de float | Gerado pelo DeepFace. Nunca sai na resposta HTTP |
| `created_at` | datetime UTC | Definido na criação |
| `updated_at` | datetime UTC | Igual a `created_at` na criação e atualizado em todo PATCH |

Índice único em `email`.

Resposta pública do usuário: `id` (ObjectId do MongoDB em string), `name`, `email`, `role`, `created_at`, `updated_at`.

## Rotas

Prefixo `/api/v1`. Upload de foto é `multipart/form-data`, campo `image`. Tipos aceitos: JPEG, PNG e WEBP. Tamanho máximo: 5 MB.

### Públicas

`POST /auth/register`

Campos: `name`, `email`, `image`. Cria o usuário com `role=user`. Resposta `201` com o usuário público.

`POST /auth/login`

Campos: `email`, `image`. Resposta `200`:

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer"
}
```

O JWT traz `sub` com o id do usuário, `role` e `exp`. Cada rota autenticada recarrega o usuário pelo `sub`. A permissão usa o `role` gravado no banco. Se o id não existir mais, a resposta é `401`.

### Autenticadas

`GET /users/me` — perfil de quem está no token.

`PATCH /users/me` — o próprio usuário, seja `user` ou `admin`. Campos opcionais: `name`, `email`, `image`. Pelo menos um campo. Foto nova substitui o embedding. Trocar o e-mail para o mesmo valor já salvo nesse usuário responde sucesso. Trocar para um e-mail de outra pessoa responde `409`.

### Somente admin

`GET /users` — lista paginada. Query `skip` (padrão 0) e `limit` (padrão 50, máximo 100). Ordem: `created_at` decrescente. Resposta: `items` e `total`.

`POST /users` — mesmo fluxo do register. Sempre grava `role=user`. Resposta `201`.

`GET /users/{id}` — um usuário.

`PATCH /users/{id}` — mesmos campos opcionais de `PATCH /users/me`.

`DELETE /users/{id}` — resposta `204`. Pode apagar o próprio admin.

`user` que chamar rota de admin recebe `403`, inclusive em `GET` ou `PATCH /users/{id}` quando o id é o próprio. O perfil próprio só passa por `/users/me`.

## Fluxo do rosto

`services/face.py` é o único lugar que chama DeepFace.

1. Ler os bytes do upload e recusar arquivo vazio, tipo fora da lista ou maior que 5 MB.
2. Decodificar para um array NumPy em BGR, em memória.
3. Chamar `DeepFace.represent` com o modelo, o detector e `enforce_detection=True`.
4. Exigir exatamente um rosto. Zero rostos, mais de um rosto ou falha do detector viram erro de requisição.
5. Devolver a lista de floats do campo `embedding`.

No login:

1. Buscar o usuário pelo e-mail normalizado. Se não existir, responder `401` sem chamar o DeepFace.
2. Extrair o embedding da foto enviada.
3. Calcular a distância de cosseno entre os dois vetores.
4. Se a distância for menor ou igual a `FACE_DISTANCE_THRESHOLD`, emitir o JWT.
5. E-mail inexistente e rosto diferente respondem o mesmo `401`, com a mensagem `Credenciais inválidas`.

Register e `POST /users` chamam o mesmo serviço de criação.

## Erros

| Situação | Status | Mensagem |
| --- | --- | --- |
| Imagem ausente | 400 | `Arquivo de imagem ausente` |
| Tipo diferente de JPEG, PNG ou WEBP | 400 | `Tipo de imagem inválido` |
| Arquivo maior que 5 MB | 400 | `Imagem maior que 5 MB` |
| Nenhum rosto | 400 | `Nenhum rosto encontrado` |
| Mais de um rosto | 400 | `Mais de um rosto encontrado` |
| Detector falha na imagem | 400 | `Não foi possível processar o rosto` |
| E-mail ou rosto não confere no login | 401 | `Credenciais inválidas` |
| Token ausente, expirado ou inválido | 401 | `Não autenticado` |
| `user` em rota de admin | 403 | `Permissão insuficiente` |
| Id inexistente nas rotas de admin | 404 | `Usuário não encontrado` |
| E-mail já cadastrado | 409 | `E-mail já cadastrado` |
| Corpo inválido | 422 | detalhe do Pydantic |

O embedding não aparece em nenhum corpo de erro.

## Testes

Os testes rodam sem MongoDB real e sem baixar pesos do DeepFace. O serviço de rosto é substituído por uma função falsa, e o repositório de usuários por um armazenamento em memória, via override de dependência do FastAPI.

Cobertura mínima:

- Register cria `user`, guarda o embedding e não o devolve.
- E-mail duplicado retorna 409.
- Imagem sem exatamente um rosto retorna 400.
- Login com distância dentro do limiar devolve JWT.
- Login com e-mail desconhecido ou distância acima do limiar retorna 401 com a mesma mensagem.
- `user` atualiza o próprio nome e uma foto nova troca o embedding.
- `user` recebe 403 em listar, criar, ler outro, atualizar outro e apagar.
- `admin` lista, cria, lê, atualiza e apaga.
- Respostas de usuário não contêm `embedding`.
