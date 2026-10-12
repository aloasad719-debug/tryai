import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = `${import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}/chat`
const STORAGE_KEY = 'my-ai-chats'

function newChat() {
  return { id: Date.now(), title: 'New chat', messages: [] }
}

function loadChats() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    if (Array.isArray(saved) && saved.length > 0) return saved
  } catch {
    /* ignore broken data */
  }
  return [newChat()]
}

function App() {
  const [chats, setChats] = useState(loadChats)
  const [activeId, setActiveId] = useState(() => loadChats()[0].id)
  const [input, setInput] = useState('')
  const [files, setFiles] = useState([])
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef(null)
  const bottomRef = useRef(null)

  const activeChat = chats.find((c) => c.id === activeId) || chats[0]
  const messages = activeChat.messages

  // Save chats in the browser (image previews are not saved, only names)
  useEffect(() => {
    const toSave = chats.map((chat) => ({
      ...chat,
      messages: chat.messages.map((m) => ({
        ...m,
        files: (m.files || []).map((f) => ({ name: f.name, type: f.type })),
      })),
    }))
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave))
    } catch {
      /* storage full - ignore */
    }
  }, [chats])

  // Scroll to newest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function updateActiveChat(updater) {
    setChats((previous) =>
      previous.map((chat) => (chat.id === activeChat.id ? updater(chat) : chat))
    )
  }

  function handleFileChange(event) {
    const selected = Array.from(event.target.files || [])
    setFiles((previous) => [...previous, ...selected])
    event.target.value = ''
  }

  function removeFile(indexToRemove) {
    setFiles((previous) => previous.filter((_, i) => i !== indexToRemove))
  }

  function startNewChat() {
    const chat = newChat()
    setChats((previous) => [chat, ...previous])
    setActiveId(chat.id)
    setSidebarOpen(false)
  }

  function deleteChat(id) {
    const remaining = chats.filter((c) => c.id !== id)
    if (remaining.length === 0) {
      const chat = newChat()
      setChats([chat])
      setActiveId(chat.id)
    } else {
      setChats(remaining)
      if (id === activeId) setActiveId(remaining[0].id)
    }
  }

  function downloadChat(chat) {
    if (chat.messages.length === 0) return
    const lines = chat.messages.map((m) => {
      const who = m.role === 'user' ? 'You' : 'My AI'
      const attached = (m.files || []).length
        ? `\n[Attached: ${m.files.map((f) => f.name).join(', ')}]`
        : ''
      return `${who}:${attached}\n${m.text}\n`
    })
    const content = `${chat.title}\n${'='.repeat(chat.title.length)}\n\n${lines.join('\n')}`
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${chat.title.replace(/[^\wঀ-৿-]+/g, '_')}.txt`
    link.click()
    URL.revokeObjectURL(link.href)
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (loading) return

    const text = input.trim()
    if (!text && files.length === 0) return

    const sentFiles = files
    const userMessage = {
      id: Date.now(),
      role: 'user',
      text,
      files: sentFiles.map((f) => ({
        name: f.name,
        type: f.type,
        url: f.type.startsWith('image/') ? URL.createObjectURL(f) : null,
      })),
    }
    const assistantId = Date.now() + 1
    const history = messages
      .filter((m) => m.text && m.text !== 'Thinking...')
      .map((m) => ({ role: m.role, text: m.text }))

    updateActiveChat((chat) => ({
      ...chat,
      title:
        chat.messages.length === 0
          ? (text || sentFiles[0].name).slice(0, 40)
          : chat.title,
      messages: [
        ...chat.messages,
        userMessage,
        { id: assistantId, role: 'assistant', text: 'Thinking...', files: [] },
      ],
    }))
    setInput('')
    setFiles([])
    setLoading(true)

    function setAnswer(answer) {
      updateActiveChat((chat) => ({
        ...chat,
        messages: chat.messages.map((m) =>
          m.id === assistantId ? { ...m, text: answer } : m
        ),
      }))
    }

    try {
      const formData = new FormData()
      formData.append('message', text)
      formData.append('history', JSON.stringify(history))
      sentFiles.forEach((f) => formData.append('files', f))

      const response = await fetch(API_URL, { method: 'POST', body: formData })
      const data = await response.json()

      if (!response.ok || data.error) {
        throw new Error(data.error || 'The server returned an error.')
      }
      setAnswer(data.reply)
    } catch (error) {
      setAnswer(`Sorry, something went wrong: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      {/* Sidebar: chat history */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-head">
          <strong>Chats</strong>
          <button
            className="icon-button"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close menu"
          >
            ×
          </button>
        </div>
        <button className="new-chat-button" onClick={startNewChat}>
          + New chat
        </button>
        <div className="chat-list">
          {chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-item ${chat.id === activeChat.id ? 'active' : ''}`}
            >
              <button
                className="chat-title"
                onClick={() => {
                  setActiveId(chat.id)
                  setSidebarOpen(false)
                }}
              >
                {chat.title}
              </button>
              <button
                className="chat-action"
                title="Download this chat"
                onClick={() => downloadChat(chat)}
              >
                ⤓
              </button>
              <button
                className="chat-action"
                title="Delete this chat"
                onClick={() => deleteChat(chat.id)}
              >
                🗑
              </button>
            </div>
          ))}
        </div>
      </aside>
      {sidebarOpen && (
        <div className="backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      <header className="topbar">
        <button
          className="icon-button menu-button"
          aria-label="Open menu"
          onClick={() => setSidebarOpen(true)}
        >
          <span>☰</span>
        </button>

        <button className="model-picker">
          My AI <span className="model-arrow"></span>
        </button>

        <button
          className="icon-button download-top"
          aria-label="Download this chat"
          title="Download this chat"
          onClick={() => downloadChat(activeChat)}
        >
          
        </button>
      </header>

      <section className={`chat-area ${messages.length ? 'has-messages' : ''}`}>
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="brand-mark">✳</div>
            <h1>What can I help you with?</h1>
            <p>Ask a question, explore an idea, or share a file.</p>
          </div>
        ) : (
          <div className="messages">
            {messages.map((message) => (
              <article
                className={
                  message.role === 'user' ? 'user-message' : 'assistant-message'
                }
                key={message.id}
              >
                {(message.files || []).length > 0 && (
                  <div className="attachment-list">
                    {message.files.map((file, index) => (
                      <div className="message-attachment" key={index}>
                        {file.url ? (
                          <img src={file.url} alt={file.name} />
                        ) : (
                          <>
                            <span className="document-icon">▤</span>
                            <span>{file.name}</span>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {message.text && <p>{message.text}</p>}
              </article>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </section>

      <footer className="composer-area">
        <form className="composer" onSubmit={handleSubmit}>
          {files.length > 0 && (
            <div className="file-previews">
              {files.map((file, index) => (
                <div className="file-preview" key={`${file.name}-${index}`}>
                  {file.type.startsWith('image/') ? (
                    <img src={URL.createObjectURL(file)} alt={file.name} />
                  ) : (
                    <span className="document-icon">▤</span>
                  )}
                  <button
                    type="button"
                    className="remove-file"
                    onClick={() => removeFile(index)}
                    aria-label={`Remove ${file.name}`}
                  >
                    ×
                  </button>
                  <span className="file-name">{file.name}</span>
                </div>
              ))}
            </div>
          )}

          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                handleSubmit(event)
              }
            }}
            placeholder="Ask anything..."
            rows={1}
            aria-label="Message"
          />

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf"
            multiple
            hidden
            onChange={handleFileChange}
          />

          <div className="composer-toolbar">
            <button
              type="button"
              className="icon-button add-button"
              onClick={() => fileInputRef.current?.click()}
              aria-label="Attach an image or PDF"
              title="Attach an image or PDF"
            >
              +
            </button>

            <div className="toolbar-right">
              <button
                type="submit"
                className="send-button"
                disabled={loading || (!input.trim() && files.length === 0)}
                aria-label="Send message"
                title="Send message"
              >
                ↑
              </button>
            </div>
          </div>
        </form>
        <p className="disclaimer">
          My AI can make mistakes. Check important information.
        </p>
      </footer>
    </main>
  )
}

export default App