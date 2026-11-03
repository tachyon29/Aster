import { useEffect, useRef, useState } from "react";
import "./App.css";

const stages = [
  { id: "planner", title: "Plan", detail: "Shape the research question", icon: "✦" },
  { id: "research", title: "Research", detail: "Check the evidence needed", icon: "⌕" },
  { id: "browser", title: "Sources", detail: "Read relevant webpages", icon: "↗" },
  { id: "answer", title: "Answer", detail: "Write a focused response", icon: "✎" },
];

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [running, setRunning] = useState(false);
  const [activeStage, setActiveStage] = useState("");
  const [stageMessage, setStageMessage] = useState("Ready when you are");
  const [finished, setFinished] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi, I’m WebPilot. What would you like to learn about?" },
  ]);
  const [graphNodes, setGraphNodes] = useState([]);
  const [currentNode, setCurrentNode] = useState(null);
  const [visitCount, setVisitCount] = useState(0);
  const messagesRef = useRef(null);
  const graphScrollRef = useRef(null);

  useEffect(() => {
    if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages]);
  useEffect(() => {
    if (graphScrollRef.current) graphScrollRef.current.scrollTop = graphScrollRef.current.scrollHeight;
  }, [graphNodes.length]);

  async function runResearch() {
    const mission = prompt.trim();
    if (!mission || running) return;

    setPrompt("");
    setRunning(true);
    setFinished(false);
    setActiveStage("planner");
    setStageMessage("Preparing research");
    setGraphNodes([]);
    setCurrentNode(null);
    setVisitCount(0);
    setMessages((current) => [
      ...current,
      { role: "user", content: mission },
      { role: "assistant", content: "Starting research…", pending: true },
    ]);

    const visitsByBase = new Map();
    let toolCycle = 0;

    const replacePending = (content) => setMessages((current) => {
      const copy = [...current];
      let index = copy.length - 1;
      while (index >= 0 && !copy[index].pending) index -= 1;
      if (index >= 0) copy[index] = { role: "assistant", content };
      return copy;
    });

    function addVisit(event) {
      const baseId = event.node_id;
      const visit = (visitsByBase.get(baseId) || 0) + 1;
      visitsByBase.set(baseId, visit);
      if (baseId === "tool_dispatch") toolCycle += 1;

      let label = event.label;
      if (baseId === "start") label = "Research request";
      else if (baseId === "research_agent") label = "Research agent · visit " + visit;
      else if (baseId === "tool_dispatch") label = "MCP dispatcher · call " + toolCycle;
      else if (baseId.startsWith("tool:")) label = event.label + " · call " + toolCycle;

      const vertex = {
        id: baseId + "@" + visit,
        baseId,
        label,
        purpose: event.purpose || "Research workflow node.",
        detail: event.detail || "",
        status: "active",
        sequence: graphNodes.length,
      };

      setGraphNodes((current) => [
        ...current.map((node) => ({ ...node, status: node.status === "active" ? "visited" : node.status })),
        { ...vertex, sequence: current.length },
      ]);
      setCurrentNode(vertex);
      setVisitCount((count) => count + 1);

    }

    try {
      const response = await fetch("http://localhost:8000/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mission }),
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Research request failed.");
      }
      if (!response.body) throw new Error("The server did not provide a progress stream.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let gotAnswer = false;
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";
        for (const frame of frames) {
          const line = frame.split("\n").find((part) => part.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === "status") {
            setActiveStage(event.stage);
            setStageMessage(event.message);
          } else if (event.type === "graph") {
            if (event.action === "reset") {
              setGraphNodes([]);
              setCurrentNode(null);
              setVisitCount(0);
            } else if (event.action === "visit") {
              addVisit(event);
            }
          } else if (event.type === "complete") {
            replacePending(event.final_output || "Research completed.");
            setFinished(true);
            setActiveStage("answer");
            setStageMessage("Answer ready");
            setGraphNodes((current) => current.map((node) => ({ ...node, status: "visited" })));
            gotAnswer = true;
          } else if (event.type === "error") {
            throw new Error(event.message || "Research failed.");
          }
        }
        if (done) break;
      }
      if (!gotAnswer) throw new Error("The research stream ended before an answer arrived.");
    } catch (error) {
      replacePending("I couldn't complete that research: " + error.message);
      setStageMessage("Research stopped");
      setFinished(false);
      setGraphNodes((current) => current.map((node) => ({ ...node, status: node.status === "active" ? "visited" : node.status })));
    } finally {
      setRunning(false);
    }
  }

  function newResearch() {
    if (running) return;
    setMessages([{ role: "assistant", content: "Hi, I’m WebPilot. What would you like to learn about?" }]);
    setPrompt("");
    setActiveStage("");
    setStageMessage("Ready when you are");
    setFinished(false);
    setGraphNodes([]);
    setCurrentNode(null);
    setVisitCount(0);
  }


  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#home" aria-label="WebPilot home">
        <span className="brand-mark">W</span><span><strong>WebPilot</strong><small>RESEARCH STUDIO</small></span>
      </a>
      <button className="new-research-btn" onClick={newResearch}><span>＋</span> New research</button>
      <div className="sidebar-label">WORKSPACE</div>
      <div className="nav-item selected"><span>◉</span> Research chat</div>
      <div className="sidebar-note"><span className="online-dot" /> Local model <b>Qwen3</b></div>
      <div className="sidebar-bottom"><span className="avatar">A</span><span><strong>Personal workspace</strong><small>Private &amp; local</small></span><span className="more">···</span></div>
    </aside>

    <main className="main-workspace">
      <header className="topbar">
        <div><div className="eyebrow">YOUR RESEARCH DESK</div><h1>Research workspace</h1><p>Ask a question. Get a clear answer grounded in the web.</p></div>
        <div className="model-pill"><span className="online-dot" /> Qwen3 <span className="pill-divider" /> On device</div>
      </header>

      <section className={"execution-panel " + (running ? "is-running" : "")}>
        <div className="execution-topline"><div><span className="eyebrow">RESEARCH IN MOTION</span><h2>{running && currentNode ? "Current node: " + currentNode.label : running ? stageMessage : finished ? "Research complete" : "Your workflow"}</h2></div><span className={"run-status " + (running ? "working" : finished ? "complete" : "ready")}><i />{running ? "WORKING" : finished ? "COMPLETE" : "READY"}</span></div>
        <div className="stage-track">
          {stages.map((stage, index) => {
            const activeIndex = stages.findIndex((item) => item.id === activeStage);
            const complete = finished || index < activeIndex;
            const active = !finished && stage.id === activeStage;
            return <div className={"stage-item " + (complete ? "complete " : "") + (active ? "active" : "")} key={stage.id}>
              <div className="stage-symbol">{complete ? "✓" : stage.icon}</div>
              <div className="stage-copy"><strong>{stage.title}</strong><span>{stage.detail}</span></div>
              {index < stages.length - 1 && <div className={"stage-connector " + (complete ? "filled" : "")} />}
            </div>;
          })}
        </div>
      </section>

      <section className="workspace-grid">
        <section className="chat-panel">
          <div className="panel-heading"><div><span className="panel-icon">✳</span><div><h2>Research chat</h2><p>Your question and answer, together</p></div></div><span className="private-label">● PRIVATE</span></div>
          <div className="chat-messages" ref={messagesRef}>
            {messages.map((message, index) => <article className={"chat-message " + message.role} key={index}>
              <div className={"chat-avatar " + (message.role === "user" ? "user-avatar" : "assistant-avatar")}>{message.role === "user" ? "A" : "W"}</div>
              <div className="message-body"><span className="message-author">{message.role === "user" ? "YOU" : "WEBPILOT"}</span><div className={"chat-bubble " + (message.pending ? "pending-bubble" : "")}>{message.content}{message.pending && <span className="typing-dots"><i /><i /><i /></span>}</div></div>
            </article>)}
          </div>
          <form className="composer-wrap" onSubmit={(event) => { event.preventDefault(); runResearch(); }}>
            <div className="composer"><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="What would you like me to research?" disabled={running} rows={2} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); runResearch(); } }} /><button className="send-btn" type="submit" aria-label="Send research request" disabled={!prompt.trim() || running}>{running ? <span className="button-spinner" /> : "↑"}</button></div>
            <div className="composer-footer"><span>Be specific for a sharper answer</span><span><kbd>Enter</kbd> to send <i /> <kbd>Shift + Enter</kbd> for a new line</span></div>
          </form>
        </section>

        <section className="graph-panel">
          <div className="panel-heading"><div><span className="panel-icon graph-icon">⌘</span><div><h2>Execution trace</h2><p>{visitCount} steps · current step is highlighted</p></div></div><span className="graph-tag">LIVE TRACE</span></div>
          <div className="graph-scroll" ref={graphScrollRef}>
            <div className="tree-canvas">
              {graphNodes.map((node, index) => <div className={"trace-row " + node.status} key={node.id}>
                <span className="trace-connector" aria-hidden="true"><b>{index + 1}</b>{index < graphNodes.length - 1 && <i />}</span>
                <div className="trace-card">
                  <div className="trace-card-heading"><strong>{node.label}</strong><small>{node.status === "active" ? "RUNNING" : node.status === "visited" ? "DONE" : "WAITING"}</small></div>
                  <p>{node.purpose}</p>
                  {node.detail && <code>{node.detail}</code>}
                </div>
              </div>)}
            </div>
          </div>
          <div className="workflow-summary"><span className="summary-pulse" /><span>{currentNode ? (running ? "Executing: " : "Last node: ") + currentNode.label : running ? stageMessage : finished ? "Answer delivered in chat" : "Waiting for a research question"}</span><span className="graph-legend"><i className="legend-active" /> Current <i className="legend-visited" /> Visited</span></div>
        </section>
      </section>
      <footer className="page-footer"><span>WebPilot researches with your local model</span><span>Steps are connected in execution order</span></footer>
    </main>
  </div>;
}
