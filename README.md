# blog_article
# Building a Production-Ready RAG System in Ruby

Retrieval-Augmented Generation (RAG) has emerged as a game-changing approach in AI applications. While Python dominates the AI landscape, Ruby offers an elegant and production-ready alternative for building RAG systems. In this article, we'll explore a complete RAG implementation in Ruby, breaking down key concepts and examining how each component works together.

## Understanding RAG Architecture

Before diving into the code, let's understand what makes our Ruby RAG implementation special:

1. **Thread-safe document store**: Handles concurrent requests safely
2. **Flexible embedding generation**: Uses OpenAI's latest embedding models
3. **Multi-provider support**: Works with both OpenAI and OpenRouter
4. **Efficient text chunking**: Implements smart text segmentation with overlap
5. **Production-ready API**: Built with Sinatra for lightweight deployment

## Core Components Deep Dive

### Smart Text Chunking

One of the most critical aspects of RAG is proper text chunking. Our implementation uses a sophisticated approach:

```ruby
def chunk_text(text, max_tokens: 800, overlap: 50)
  sentences = text.split(/(?<=[.!?])\s+/)
  chunks = []
  current_chunk = []
  current_tokens = 0

  sentences.each do |sentence|
    sentence_tokens = TOKENIZER.encode(sentence).tokens.size
    
    if current_tokens + sentence_tokens > max_tokens
      chunks << build_chunk(current_chunk, overlap)
      current_chunk = current_chunk.last(overlap)
      current_tokens = current_chunk.sum { |s| TOKENIZER.encode(s).tokens.size }
    end
    
    current_chunk << sentence
    current_tokens += sentence_tokens
  end
  
  chunks << build_chunk(current_chunk, overlap) unless current_chunk.empty?
  chunks
end
```

This chunking algorithm:
- Preserves sentence boundaries for natural context
- Maintains overlap between chunks to preserve context
- Respects token limits while maximizing content

### Thread-Safe Document Store

The DocumentStore class implements thread safety using Ruby's Monitor module:

```ruby
class DocumentStore
  include MonitorMixin

  def initialize
    super
    @documents = []
    @index = []
  end

  def add(chunk, embedding)
    synchronize do
      @documents << { chunk: chunk, embedding: embedding }
      @index << embedding
    end
  end
end
```

This ensures our RAG system can handle multiple concurrent requests without data corruption - essential for production deployments.

### Intelligent Response Generation

The RAGBrain class orchestrates the response generation with a carefully crafted system prompt:

```ruby
SYSTEM_PROMPT = <<~SYS.freeze
  Eres un asistente especializado en análisis de documentos. Respuestas deben:
  - Basarse exclusivamente en el contexto proporcionado
  - Ser concisas (máximo 3 oraciones)
  - Incluir referencias tipo [1] cuando aplique
  - Indicar claramente cuando no haya información suficiente

  Contexto disponible:
  %<context>s
SYS
```

This prompt engineering ensures:
- Responses are grounded in provided context
- Clear attribution through references
- Concise and focused answers
- Transparency about information gaps

## Production Considerations

### Error Handling and Fallbacks

The implementation includes robust error handling throughout:

```ruby
def embed(text)
  return @cache[text] if @cache.key?(text)
  
  clean_text = text.truncate(MAX_TOKENS * 4, omission: "")
  response = @client.embeddings(
    parameters: {
      model: MODEL_EMBEDDINGS,
      input: clean_text
    }
  )

  raise "Embedding error: #{response.dig("error", "message")}" if response["error"]

  Vector[*response.dig("data", 0, "embedding")].tap do |vec|
    @cache[text] = vec
  end
rescue => e
  puts "Embedding fallback: #{e.message}"
  Vector.elements(Array.new(1536, 0.0))
end
```

Key features include:
- Caching for performance
- Fallback vectors for error cases
- Clear error messaging
- Token limit handling

### API Design

The Sinatra API provides a clean interface:

```ruby
post "/ingest" do
  text = parse_text(request)
  emb_client = EmbeddingClient.new
  
  chunks = emb_client.chunk_text(text)
  chunks.each { |chunk| DOCUMENT_STORE.add(chunk, emb_client.embed(chunk)) }

  { status: "success", chunks: chunks.size }.to_json
end

post "/ask" do
  data = JSON.parse(request.body.read)
  question = data["query"].to_s.strip
  halt 400, "Query requerida" if question.empty?

  emb_client = EmbeddingClient.new
  embedding = emb_client.embed(question)
  results = DOCUMENT_STORE.search(embedding)
  
  generate_response(question, results)
end
```

This API design:
- Validates inputs thoroughly
- Provides meaningful error responses
- Returns detailed response metadata
- Maintains RESTful principles

## Deployment Tips

To deploy this RAG system in production:

1. **Environment Configuration**:
   ```ruby
   ENV["OPENAI_API_KEY"] ||= ENV.fetch("OPENAI_ACCESS_TOKEN", nil)
   ENV["OR_ACCESS_TOKEN"] ||= ENV.fetch("OPENROUTER_API_KEY", nil)
   ```
   Use environment variables for sensitive configuration.

2. **Memory Management**:
   The stats endpoint helps monitor system health:
   ```ruby
   get "/stats" do
     {
       documents: DOCUMENT_STORE.size,
       memory: "%d MB" % (`ps -o rss= -p #{Process.pid}`.to_i / 1024)
     }.to_json
   end
   ```

3. **Scaling Considerations**:
   - Use Redis or PostgreSQL for the document store in larger deployments
   - Implement rate limiting for API endpoints
   - Consider background job processing for document ingestion

## Conclusion

Ruby proves to be an excellent choice for building production-ready RAG systems. The language's elegant syntax and robust concurrency support, combined with powerful gems like Sinatra, create a solid foundation for AI applications.

This implementation demonstrates that you don't need complex Python frameworks to build sophisticated RAG systems. Ruby's simplicity and productivity shine through, making it an excellent choice for teams looking to integrate RAG into their existing Ruby applications.

Remember to check the full source code for additional features and optimizations not covered in this article. The modular design makes it easy to extend and customize for your specific needs.
