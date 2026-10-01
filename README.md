# Assistente Acadêmico IFPA com RAG e Agente ReAct

Protótipo acadêmico que responde perguntas sobre normas e documentos do IFPA. A solução carrega PDFs oficiais, divide o texto em fragmentos, gera representações vetoriais, persiste o índice no ChromaDB e oferece a busca como Tool para um agente ReAct.

## Documentos incluídos

- Resolução CONSUP/IFPA nº 944/2023, aplicável à graduação.
- Resolução CONSUP/IFPA nº 945/2023, aplicável à educação básica e profissional.
- PPC do Bacharelado em Ciência da Computação do Campus Ananindeua.
- Calendário de graduação de 2026 do Campus Ananindeua.

## Execução recomendada no Google Colab

1. Abra `Assistente_Academico_IFPA.ipynb` no Colab.
2. Envie a pasta `documentos` e o arquivo `src/assistente_ifpa.py`, preservando a estrutura do projeto.
3. No painel Secrets, crie `GEMINI_API_KEY` e autorize o notebook.
4. Execute as células em ordem.
5. Para a defesa, mantenha o diretório `chroma_db_ifpa` salvo no Drive.

Sem chave de API, o projeto ativa embeddings locais determinísticos e uma resposta extrativa. Esse modo permite demonstrar indexação, recuperação e citação de fontes sem consumir cota. O caminho principal avaliado utiliza Gemini, ChromaDB e o agente ReAct.

## Segurança e resiliência

- Nenhuma chave é gravada no código.
- Consultas com padrões comuns de prompt injection são bloqueadas antes da chamada ao modelo.
- O agente recebe limite de recursão e só responde com contexto recuperado.
- As fontes e páginas são anexadas aos trechos recuperados.
- Erros de API ativam o modo local quando possível.

## Estrutura

```text
miniprojeto/
  Assistente_Academico_IFPA.ipynb
  documentos/
  entregas/
  src/assistente_ifpa.py
  tests/test_logic.py
  requirements.txt
  .env.example
```

## Testes locais

```powershell
python -m unittest discover -s tests -v
```

## Materiais da defesa

A pasta `entregas` contém o relatório em PDF e DOCX, a apresentação final, o roteiro do pitch e a ficha de coavaliação. Antes da submissão oficial, substitua os campos `[GRUPO X]`, `[NOME]` e `[MATRÍCULA]`.

## Fontes oficiais

- https://proen.ifpa.edu.br/documentos-1/regulamento-didatico-de-ensino
- https://ananindeua.ifpa.edu.br/component/content/article/2-uncategorised/629-curso-graduacao
- https://ananindeua.ifpa.edu.br/ultimas-noticias/1292-calendario-2026

