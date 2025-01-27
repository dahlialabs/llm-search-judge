
## Local LLM Search Relevance Judge

(Runs on Apple Silicon only with MLX)

See [original README](https://github.com/softwaredoug/local-llm-judge/blob/main/README.md).

### To eval search results

With poetry deps installed

```
source ./connect.sh    # Annoying connect to Elasticsearch to get category
poetry run jupyter notebook
```

Pasted in queries, see side by side relevanec preferences
