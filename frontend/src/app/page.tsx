"use client";

import { useEffect, useRef, useState } from "react";
import { io, Socket } from "socket.io-client";

const socket: Socket = io("http://localhost:8000");

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [message, setMessage] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [connected, setConnected] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // ─────────────────────────────────────────────────────────────
  // Auto scroll
  // ─────────────────────────────────────────────────────────────

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // ─────────────────────────────────────────────────────────────
  // Socket Events
  // ─────────────────────────────────────────────────────────────

  useEffect(() => {
    socket.on("connect", () => {
      setConnected(true);
    });

    socket.on("disconnect", () => {
      setConnected(false);
    });

    socket.on("token", (data: { token: string }) => {
  setMessages((prev) => {
    const updated = [...prev];

    if (
      updated.length > 0 &&
      updated[updated.length - 1].role === "assistant"
    ) {
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        content:
          updated[updated.length - 1].content +
          data.token,
      };
    }

    return updated;
  });
});

    socket.on("sources", (data: { sources: string[] }) => {
      setSources(data.sources);
    });

    socket.on("done", () => {
      setLoading(false);
    });

    socket.on("error", (data: { message: string }) => {
      setLoading(false);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${data.message}`,
        },
      ]);
    });

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("token");
      socket.off("sources");
      socket.off("done");
      socket.off("error");
    };
  }, []);

  // ─────────────────────────────────────────────────────────────
  // Send Message
  // ─────────────────────────────────────────────────────────────

  const sendMessage = () => {
    if (!message.trim()) return;

    setSources([]);
    setLoading(true);

    // Add user message
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        content: message,
      },
      {
        role: "assistant",
        content: "",
      },
    ]);

    socket.emit("message", {
      message,
    });

    setMessage("");
  };

  // ─────────────────────────────────────────────────────────────
  // Enter key support
  // ─────────────────────────────────────────────────────────────

  const handleKeyDown = (
    e: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  // ─────────────────────────────────────────────────────────────
  // UI
  // ─────────────────────────────────────────────────────────────

  return (
    <main className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="border-b border-gray-800 px-8 py-6">
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-bold">
            Gemma Voice Agent
          </h1>

          <div
            className={`text-sm px-3 py-1 rounded-full ${
              connected
                ? "bg-green-900 text-green-300"
                : "bg-red-900 text-red-300"
            }`}
          >
            {connected ? "Connected" : "Disconnected"}
          </div>
        </div>

        <p className="text-gray-400 mt-2">
          Multilingual RAG Assistant powered by Gemma + ChromaDB
        </p>
      </div>

      {/* Chat Area */}
      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="space-y-6">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${
                msg.role === "user"
                  ? "justify-end"
                  : "justify-start"
              }`}
            >
              <div
                className={`max-w-3xl px-5 py-4 rounded-2xl whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "bg-white text-black"
                    : "bg-zinc-900 border border-zinc-800"
                }`}
              >
                <div className="text-xs opacity-60 mb-2">
                  {msg.role === "user"
                    ? "You"
                    : "Assistant"}
                </div>

                {msg.content || (
                  <span className="animate-pulse">
                    Thinking...
                  </span>
                )}
              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        {/* Sources */}
        {sources.length > 0 && (
          <div className="mt-10 border border-zinc-800 rounded-2xl p-6 bg-zinc-950">
            <h2 className="text-xl font-bold mb-4">
              Sources
            </h2>

            <div className="flex flex-wrap gap-3">
              {sources.map((src, idx) => (
                <div
                  key={idx}
                  className="bg-zinc-900 border border-zinc-700 px-4 py-2 rounded-lg text-sm"
                >
                  {src}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="fixed bottom-0 left-0 w-full border-t border-zinc-800 bg-black">
        <div className="max-w-5xl mx-auto px-6 py-5">
          <div className="flex gap-4">
            <input
              className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl px-5 py-4 outline-none focus:border-white"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask something in English / Hindi / Bengali..."
            />

            <button
              className="bg-white text-black px-8 py-4 rounded-xl font-semibold hover:opacity-90 transition"
              onClick={sendMessage}
              disabled={loading}
            >
              {loading ? "..." : "Send"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}