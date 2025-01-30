# mini_rag.rb
require "matrix"
require "sinatra"
require "raix"
require "json"
require "openai"
require "tokenizers"
require "monitor"
require "open_router"

# ---------------------------------------------------------------
# Configuración mejorada con validación y soporte múltiple clients
# ---------------------------------------------------------------

# Configuración de variables de entorno
ENV["OPENAI_API_KEY"] ||= ENV.fetch("OPENAI_ACCESS_TOKEN", nil)
ENV["OR_ACCESS_TOKEN"] ||= ENV.fetch("OPENROUTER_API_KEY", nil)

# Configuración del módulo Raix
module Raix
  class << self
    attr_accessor :configuration
  end

  def self.configure
    self.configuration ||= Configuration.new
    yield(configuration)
  end

  class Configuration
    attr_accessor :openrouter_client, :openai_client
  end
end

# Inicialización de clientes
Raix.configure do |config|
  if ENV["OPENAI_API_KEY"]
    config.openai_client = OpenAI::Client.new(access_token: ENV["OPENAI_API_KEY"])
  end
  
  if ENV["OR_ACCESS_TOKEN"]
    config.openrouter_client = OpenRouter::Client.new(
      access_token: ENV["OR_ACCESS_TOKEN"],
      site_url: "http://localhost:4567",
      site_name: "MiniRAG"
    )
  end
end

# Extensión de la clase String para el método truncate
class String
  def truncate(length, omission: "...")
    return self if size <= length
    "#{self[0, length - omission.length]}#{omission}"
  end
end

# ---------------------------------------------------------------
# Cliente de Embeddings mejorado
# ---------------------------------------------------------------
class EmbeddingClient
  MODEL_EMBEDDINGS = "text-embedding-3-small"
  MAX_TOKENS = 8191
  TOKENIZER = Tokenizers.from_pretrained("gpt2")

  def initialize
    @client = OpenAI::Client.new(access_token: ENV["OPENAI_API_KEY"])
    @cache = {}
  end

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

  private

  def build_chunk(sentences, overlap)
    return sentences.join(" ") if sentences.size <= overlap
    sentences.join(" ")
  end
end

# ---------------------------------------------------------------
# Almacén de documentos thread-safe
# ---------------------------------------------------------------
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

  def search(query_embedding, top_k: 3)
    synchronize do
      return [] if @documents.empty?
      
      scores = @index.map { |vec| vec.inner_product(query_embedding) }
      top_indices = scores.each_with_index.max_by(top_k) { |score, _| score }.map(&:last)
      top_indices.map { |i| @documents[i] }
    end
  end

  def size
    synchronize { @documents.size }
  end
end

DOCUMENT_STORE = DocumentStore.new

# ---------------------------------------------------------------
# Motor RAG con soporte para OpenAI y OpenRouter
# ---------------------------------------------------------------
class RAGBrain
  include Raix::ChatCompletion

  SYSTEM_PROMPT = <<~SYS.freeze
    Eres un asistente especializado en análisis de documentos. Respuestas deben:
    - Basarse exclusivamente en el contexto proporcionado
    - Ser concisas (máximo 3 oraciones)
    - Incluir referencias tipo [1] cuando aplique
    - Indicar claramente cuando no haya información suficiente

    Contexto disponible:
    %<context>s
  SYS

  def initialize(question, context)
    @question = question
    @context = process_context(context)
    build_transcript
  end

  def generate_response
    begin
      if ENV["OR_ACCESS_TOKEN"]
        client = Raix.configuration.openrouter_client
        result = client.complete(
          messages,
          model: ["mistralai/mistral-7b-instruct", "openai/gpt-3.5-turbo"],
          extras: {
            temperature: 0.2,
            max_tokens: 150
          }
        )
      else
        client = Raix.configuration.openai_client
        result = client.chat(
          parameters: {
            model: "gpt-3.5-turbo",
            messages: messages,
            temperature: 0.2,
            max_tokens: 150
          }
        )
      end

      response = result.dig("choices", 0, "message", "content")
      validate_response(response)
    rescue => e
      "Error generando respuesta: #{e.message}"
    end
  end

  private

  def process_context(chunks)
    return "" if chunks.empty?
    
    chunks_array = chunks.is_a?(Array) ? chunks : [chunks]
    
    chunks_array.map.with_index(1) do |chunk, i|
      chunk_text = chunk.is_a?(String) ? chunk : chunk[:chunk].to_s
      "Fragmento [#{i}]: #{chunk_text[0..200]}"
    end.join("\n")
  end

  def build_transcript
    @messages = [
      {
        role: "system",
        content: format(SYSTEM_PROMPT, context: @context)
      },
      {
        role: "user",
        content: @question
      }
    ]
  end

  def messages
    @messages ||= []
  end

  def validate_response(response)
    return "No hay información suficiente en los documentos para responder esta pregunta." if response.nil?
    
    if response.match?(/no (tengo|dispongo|hay información)/i)
      "No hay información suficiente en los documentos para responder esta pregunta."
    else
      response.gsub(/\[(\d+)\]/) { |m| "[#{m[1].to_i + 1}]" }
    end
  end
end

# ---------------------------------------------------------------
# API Sinatra
# ---------------------------------------------------------------
class MiniRAGApp < Sinatra::Base
  configure do
    set :port, 4567
    set :environment, :production
    set :show_exceptions, false
  end

  before do
    content_type :json
  end

  error 400..500 do
    { error: env["sinatra.error"].message }.to_json
  end

  get "/" do
    content_type :html
    <<~HTML
      <h1>miniRAG Avanzado</h1>
      <p>Soporta OpenAI y OpenRouter</p>
      <ul>
        <li>POST /ingest - Texto a indexar</li>
        <li>POST /ask - Consulta RAG</li>
        <li>GET /stats - Estado del sistema</li>
      </ul>
    HTML
  end

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

  get "/stats" do
    {
      documents: DOCUMENT_STORE.size,
      memory: "%d MB" % (`ps -o rss= -p #{Process.pid}`.to_i / 1024)
    }.to_json
  end

  private

  def parse_text(request)
    text = if request.content_type == "application/json"
             JSON.parse(request.body.read)["text"]
           else
             params["text"]
           end.to_s.strip
    
    text.empty? ? raise(ArgumentError, "Texto requerido") : text
  end

  def generate_response(question, documents)
    docs_array = documents.is_a?(Array) ? documents : [documents]
    
    context = docs_array.map { |d| d[:chunk].to_s }.join("\n")
    brain = RAGBrain.new(question, context)
    
    {
      query: question,
      answer: brain.generate_response,
      context_used: docs_array.size,
      context_preview: docs_array.first(3).map { |d| d[:chunk].to_s.truncate(50) }
    }.to_json
  end
end

# Iniciar la aplicación
MiniRAGApp.run!