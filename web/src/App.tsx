import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'

type VectorSearchResult = {
  content: string
  source: string
  page: string | number
  relevance_score: number
}

type ChatTurn = {
  id: string
  role: 'user' | 'assistant'
  content: string
}

type StreamEvent =
  | { type: 'tool_call'; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; content: string }
  | { type: 'final_answer'; content: string }
  | { type: 'error'; content: string }
  | { type: 'done' }

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

function App() {
  const [activeView, setActiveView] = useState<'chat' | 'knowledge' | 'vector'>('chat')
  const [documents, setDocuments] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')

  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [chatting, setChatting] = useState(false)
  const [trace, setTrace] = useState<string[]>([])
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([])
  const [sessionId, setSessionId] = useState('')
  const [sessionMessage, setSessionMessage] = useState('')

  const [stats, setStats] = useState<{
    collection_name: string
    persist_dir: string
    total_chunks: number
    source_count: number
    top_sources: Array<{ source: string; chunks: number }>
  } | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<VectorSearchResult[]>([])
  const [searching, setSearching] = useState(false)

  const statusText = useMemo(() => {
    if (!sessionId) return 'Creating session...'
    if (chatting) return 'Running agent...'
    if (ingesting) return 'Ingestion running...'
    if (uploading) return 'Uploading files...'
    return 'Ready'
  }, [chatting, ingesting, sessionId, uploading])

  useEffect(() => {
    void createSession()
    void refreshDocuments()
    void refreshStats()
  }, [])

  async function createSession() {
    try {
      const res = await fetch(`${API_BASE}/sessions`, { method: 'POST' })
      const data = await res.json()
      setSessionId(data.session_id ?? '')
      setSessionMessage('Session ready.')
    } catch (err) {
      setSessionId('')
      setSessionMessage(`Session creation failed: ${String(err)}`)
    }
  }

  async function resetSession() {
    if (sessionId) {
      try {
        await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' })
      } catch {
        // no-op: reset path should continue even if delete fails
      }
    }

    setQuestion('')
    setAnswer('')
    setTrace([])
    setChatTurns([])
    await createSession()
  }

  async function refreshDocuments() {
    const res = await fetch(`${API_BASE}/documents`)
    const data = await res.json()
    setDocuments(data.documents ?? [])
  }

  async function refreshStats() {
    try {
      const res = await fetch(`${API_BASE}/vector/stats`)
      const data = await res.json()
      setStats(data)
    } catch {
      setStats(null)
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const input = document.getElementById('upload-files') as HTMLInputElement | null
    const files = input?.files
    if (!files || files.length === 0) {
      setUploadMessage('Select at least one PDF, MD, or TXT file.')
      return
    }

    const formData = new FormData()
    for (const file of files) formData.append('files', file)

    setUploading(true)
    setUploadMessage('')
    try {
      const res = await fetch(`${API_BASE}/ingest/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      setUploadMessage(`Saved ${data.saved_count ?? 0} file(s).`)
      await refreshDocuments()
    } catch (err) {
      setUploadMessage(`Upload failed: ${String(err)}`)
    } finally {
      setUploading(false)
      if (input) input.value = ''
    }
  }

  async function runIngestion() {
    setIngesting(true)
    setUploadMessage('')
    try {
      const res = await fetch(`${API_BASE}/ingest/run`, { method: 'POST' })
      const data = await res.json()
      setUploadMessage(data.message ?? 'Ingestion completed.')
      await refreshStats()
    } catch (err) {
      setUploadMessage(`Ingestion failed: ${String(err)}`)
    } finally {
      setIngesting(false)
    }
  }

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!question.trim() || chatting || !sessionId) return

    setChatting(true)
    setAnswer('')
    setTrace([])
    const askedQuestion = question.trim()
    setChatTurns((prev) => [
      ...prev,
      { id: `${Date.now()}-user`, role: 'user', content: askedQuestion },
    ])
    setQuestion('')

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: askedQuestion, session_id: sessionId }),
      })
      if (!response.body) {
        throw new Error('No streaming body returned by API')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder('utf-8')
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''

        for (const part of parts) {
          const line = part
            .split('\n')
            .find((chunk) => chunk.startsWith('data: '))
          if (!line) continue

          const payload = line.slice(6)
          const eventData = JSON.parse(payload) as StreamEvent

          if (eventData.type === 'tool_call') {
            setTrace((prev) => [...prev, `Tool call: ${eventData.name} ${JSON.stringify(eventData.args)}`])
          }
          if (eventData.type === 'tool_result') {
            setTrace((prev) => [...prev, `Tool result: ${eventData.content}`])
          }
          if (eventData.type === 'final_answer') {
            setAnswer(eventData.content)
            setChatTurns((prev) => [
              ...prev,
              { id: `${Date.now()}-assistant`, role: 'assistant', content: eventData.content },
            ])
          }
          if (eventData.type === 'error') {
            setTrace((prev) => [...prev, `Error: ${eventData.content}`])
          }
        }
      }
    } catch (err) {
      setAnswer(`Request failed: ${String(err)}`)
    } finally {
      setChatting(false)
    }
  }

  async function runVectorSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!searchQuery.trim()) return

    setSearching(true)
    try {
      const res = await fetch(`${API_BASE}/vector/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery.trim(), top_k: 5 }),
      })
      const data = await res.json()
      setSearchResults(data.results ?? [])
    } catch {
      setSearchResults([])
    } finally {
      setSearching(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="kicker">LOCAL REACT RAG CONTROL ROOM</p>
          <h1>Knowledge Ops Console</h1>
          {sessionId && <p className="kicker">Session: {sessionId.slice(0, 8)}...</p>}
        </div>
        <div className="status-pill">{statusText}</div>
      </header>

      <nav className="tabs">
        <button className={activeView === 'chat' ? 'active' : ''} onClick={() => setActiveView('chat')}>
          Chat
        </button>
        <button
          className={activeView === 'knowledge' ? 'active' : ''}
          onClick={() => setActiveView('knowledge')}
        >
          Knowledge Base
        </button>
        <button
          className={activeView === 'vector' ? 'active' : ''}
          onClick={() => setActiveView('vector')}
        >
          Vector DB
        </button>
      </nav>

      {activeView === 'chat' && (
        <section className="panel">
          <h2>Ask Questions</h2>
          <div className="actions-row">
            <button onClick={() => void resetSession()} disabled={chatting}>
              Reset Session
            </button>
            {sessionMessage && <p className="muted">{sessionMessage}</p>}
          </div>
          <form onSubmit={askQuestion} className="stack">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about your uploaded documents..."
              rows={4}
            />
            <button type="submit" disabled={chatting || !sessionId}>
              {chatting ? 'Running...' : 'Run ReAct Agent'}
            </button>
          </form>

          <section className="chat-log">
            {chatTurns.length === 0 ? (
              <p className="muted">Start a conversation. The assistant will remember the current session.</p>
            ) : (
              chatTurns.map((turn) => (
                <article key={turn.id} className={`bubble ${turn.role}`}>
                  <p className="bubble-role">{turn.role === 'user' ? 'You' : 'Assistant'}</p>
                  <p>{turn.content}</p>
                </article>
              ))
            )}
          </section>

          <div className="split-grid">
            <article>
              <h3>Answer</h3>
              <pre>{answer || 'No answer yet.'}</pre>
            </article>
            <article>
              <h3>Agent Trace</h3>
              <ul className="trace-list">
                {trace.length === 0 ? <li>No trace yet.</li> : trace.map((item, idx) => <li key={idx}>{item}</li>)}
              </ul>
            </article>
          </div>
        </section>
      )}

      {activeView === 'knowledge' && (
        <section className="panel">
          <h2>Upload and Ingest</h2>
          <form onSubmit={handleUpload} className="stack">
            <input id="upload-files" type="file" multiple accept=".pdf,.md,.txt" />
            <button type="submit" disabled={uploading}>
              {uploading ? 'Uploading...' : 'Upload Files'}
            </button>
          </form>

          <div className="actions-row">
            <button onClick={runIngestion} disabled={ingesting}>
              {ingesting ? 'Ingesting...' : 'Run Ingestion'}
            </button>
            <button onClick={() => void refreshDocuments()}>Refresh Files</button>
          </div>

          {uploadMessage && <p className="notice">{uploadMessage}</p>}

          <h3>Current Documents</h3>
          <ul className="doc-list">
            {documents.length === 0 ? <li>No files uploaded yet.</li> : documents.map((doc) => <li key={doc}>{doc}</li>)}
          </ul>
        </section>
      )}

      {activeView === 'vector' && (
        <section className="panel">
          <div className="stats-grid">
            <article>
              <h3>Collection</h3>
              <p>{stats?.collection_name ?? 'N/A'}</p>
            </article>
            <article>
              <h3>Total Chunks</h3>
              <p>{stats?.total_chunks ?? 0}</p>
            </article>
            <article>
              <h3>Sources</h3>
              <p>{stats?.source_count ?? 0}</p>
            </article>
          </div>

          <p className="muted">Persist dir: {stats?.persist_dir ?? 'N/A'}</p>

          <h3>Top Sources</h3>
          <ul className="doc-list">
            {(stats?.top_sources ?? []).map((item) => (
              <li key={item.source}>
                {item.source} <span>{item.chunks} chunks</span>
              </li>
            ))}
          </ul>

          <h3>Semantic Search Inspector</h3>
          <form onSubmit={runVectorSearch} className="stack">
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search vector index..."
            />
            <button type="submit" disabled={searching}>
              {searching ? 'Searching...' : 'Search'}
            </button>
          </form>

          <div className="search-results">
            {searchResults.length === 0 ? (
              <p className="muted">No results yet.</p>
            ) : (
              searchResults.map((result, idx) => (
                <article key={idx}>
                  <h4>
                    {result.source} | Page {String(result.page)} | Score {result.relevance_score}
                  </h4>
                  <p>{result.content}</p>
                </article>
              ))
            )}
          </div>
        </section>
      )}
    </div>
  )
}

export default App
