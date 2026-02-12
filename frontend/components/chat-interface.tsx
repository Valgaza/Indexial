"use client"

import React from "react"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  Send,
  Loader2,
  Trash2,
  MessageSquare,
  Code2,
  Sparkles,
  Copy,
  Check,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import type { ChatMessage } from "@/lib/types"
import { sendQuery, clearSession, resetDatabaseBeacon } from "@/lib/api"

function generateSessionId() {
  return `session_${Math.random().toString(36).substring(2, 15)}_${Date.now()}`
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="group relative mt-2 overflow-hidden rounded-lg border border-[hsl(225,12%,18%)] bg-[hsl(228,15%,6%)]">
      <div className="flex items-center justify-between border-b border-[hsl(225,12%,18%)] px-3 py-1.5">
        <span className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground">
          <Code2 className="h-3 w-3" />
          SQL
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
        >
          {copied ? (
            <Check className="h-3 w-3" />
          ) : (
            <Copy className="h-3 w-3" />
          )}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3">
        <code className="font-mono text-xs leading-relaxed text-emerald-400">
          {code}
        </code>
      </pre>
    </div>
  )
}

const routeConfig = {
  SQL: {
    color: "bg-[hsl(224,76%,56%)]/20 text-[hsl(224,76%,70%)] border-[hsl(224,76%,56%)]/30",
    label: "SQL",
  },
  RAG: {
    color: "bg-pink-500/20 text-pink-400 border-pink-500/30",
    label: "RAG",
  },
  HYBRID: {
    color: "bg-indigo-500/20 text-indigo-400 border-indigo-500/30",
    label: "HYBRID",
  },
}

function ChatMessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user"

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] lg:max-w-[75%] ${
          isUser
            ? "rounded-2xl rounded-br-md bg-[hsl(224,76%,56%)] px-4 py-2.5 text-[hsl(0,0%,100%)]"
            : "glass rounded-2xl rounded-bl-md px-4 py-3"
        }`}
      >
        {!isUser && message.route && (
          <div className="mb-2 flex items-center gap-2">
            <Badge
              className={`${routeConfig[message.route].color} border px-1.5 py-0 text-[10px]`}
            >
              {routeConfig[message.route].label}
            </Badge>
            {message.rewritten_query && (
              <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                <Sparkles className="h-2.5 w-2.5" />
                Rewritten
              </span>
            )}
          </div>
        )}

        {!isUser && message.rewritten_query && (
          <div className="mb-2 rounded-md bg-[hsl(270,60%,58%)]/10 px-2.5 py-1.5">
            <p className="text-[10px] font-medium text-[hsl(270,60%,70%)]">
              Rewritten query:
            </p>
            <p className="text-xs text-[hsl(270,60%,75%)]">
              {message.rewritten_query}
            </p>
          </div>
        )}

        <p
          className={`text-sm leading-relaxed ${isUser ? "text-[hsl(0,0%,100%)]" : "text-foreground"}`}
        >
          {message.content}
        </p>

        {!isUser && message.sql && (
          <CodeBlock code={message.sql} />
        )}

        <p
          className={`mt-1.5 text-[10px] ${isUser ? "text-[hsl(224,76%,85%)]" : "text-muted-foreground"}`}
        >
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </div>
    </div>
  )
}

export function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(generateSessionId)
  const [clearDialogOpen, setClearDialogOpen] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      const viewport = scrollRef.current.querySelector(
        "[data-radix-scroll-area-viewport]",
      )
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight
      }
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  useEffect(() => {
    const handleBeforeUnload = () => {
      resetDatabaseBeacon()
    }
    window.addEventListener("beforeunload", handleBeforeUnload)
    return () => window.removeEventListener("beforeunload", handleBeforeUnload)
  }, [])

  const handleSend = async () => {
    const trimmed = input.trim()
    if (!trimmed || isLoading) return

    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await sendQuery(trimmed, sessionId)
      const assistantMessage: ChatMessage = {
        id: `msg_${Date.now()}_assistant`,
        role: "assistant",
        content: response.answer,
        route: response.route,
        sql: response.sql,
        sql_explanation: response.sql_explanation,
        tables_used: response.tables_used,
        rewritten_query: response.rewritten_query,
        sources: response.sources,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: `msg_${Date.now()}_error`,
        role: "assistant",
        content:
          err instanceof Error
            ? `Error: ${err.message}. Please check your connection and try again.`
            : "An unexpected error occurred. Please try again.",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleClear = async () => {
    try {
      await clearSession(sessionId)
    } catch {
      // Continue clearing locally even if API call fails
    }
    setMessages([])
    setSessionId(generateSessionId())
    setClearDialogOpen(false)
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="glass-strong flex items-center justify-between px-4 py-3 lg:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[hsl(224,76%,56%)]/20">
            <MessageSquare className="h-4 w-4 text-[hsl(224,76%,56%)]" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-foreground">Chat</h1>
            <p className="text-[10px] text-muted-foreground">
              Session: {sessionId.substring(0, 20)}...
            </p>
          </div>
        </div>

        <Dialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
          <DialogTrigger asChild>
            <button
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-red-500/10 hover:text-red-400 disabled:pointer-events-none disabled:opacity-50"
              disabled={messages.length === 0}
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Clear</span>
            </button>
          </DialogTrigger>
          <DialogContent className="glass-strong border-[hsl(225,12%,22%)] sm:max-w-md">
            <DialogHeader>
              <DialogTitle className="text-foreground">
                Clear chat history?
              </DialogTitle>
              <DialogDescription>
                This will remove all messages and create a new session. This
                action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="gap-2">
              <Button
                variant="outline"
                onClick={() => setClearDialogOpen(false)}
                className="border-[hsl(225,12%,22%)] bg-transparent text-foreground hover:bg-secondary"
              >
                Cancel
              </Button>
              <Button
                onClick={handleClear}
                className="bg-red-500/20 text-red-400 hover:bg-red-500/30"
              >
                Clear history
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Messages */}
      <ScrollArea ref={scrollRef} className="flex-1 px-4 py-4 lg:px-6">
        {messages.length === 0 ? (
          <div className="flex h-full min-h-[300px] flex-col items-center justify-center gap-4">
            <div className="rounded-2xl bg-[hsl(224,76%,56%)]/10 p-4">
              <MessageSquare className="h-8 w-8 text-[hsl(224,76%,56%)]/60" />
            </div>
            <div className="text-center">
              <h3 className="text-sm font-medium text-foreground">
                Start a conversation
              </h3>
              <p className="mt-1 max-w-xs text-xs text-muted-foreground">
                Upload documents and ask questions about your data. Queries are
                automatically routed to SQL, RAG, or hybrid search.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                "How many tables are in the database?",
                "Show me 5 rows from the first table",
                "What columns does the somatization table have?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => setInput(q)}
                  className="glass-subtle rounded-full px-3 py-1.5 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {messages.map((msg) => (
              <ChatMessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="glass flex items-center gap-2 rounded-2xl rounded-bl-md px-4 py-3">
                  <Loader2 className="h-4 w-4 animate-spin text-[hsl(224,76%,56%)]" />
                  <span className="text-sm text-muted-foreground">
                    Thinking...
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="p-4 pt-0 lg:px-6">
        <div className="glass-strong flex items-end gap-2 rounded-xl p-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your data..."
            disabled={isLoading}
            rows={1}
            className="max-h-32 min-h-[36px] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-foreground placeholder-muted-foreground outline-none disabled:opacity-50"
            style={{
              height: "auto",
              overflow: "hidden",
            }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement
              target.style.height = "auto"
              target.style.height = `${Math.min(target.scrollHeight, 128)}px`
              target.style.overflow =
                target.scrollHeight > 128 ? "auto" : "hidden"
            }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[hsl(224,76%,56%)] text-[hsl(0,0%,100%)] transition-all hover:bg-[hsl(224,76%,48%)] disabled:opacity-40 disabled:hover:bg-[hsl(224,76%,56%)]"
            aria-label="Send message"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
