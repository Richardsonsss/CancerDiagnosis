import { useEffect, useRef, useState } from "react";
import { AuthError, Diagnosis, Task, diagnose, fetchTasks, getToken, preparePhoto, setToken } from "./api";

type Status =
  | { kind: "idle" }
  | { kind: "busy" }
  | { kind: "done"; result: Diagnosis }
  | { kind: "error"; message: string };

export default function App() {
  const [tasks, setTasks] = useState<Task[] | null>(null);
  const [taskId, setTaskId] = useState("");
  const [needToken, setNeedToken] = useState(false);
  const [tokenInput, setTokenInput] = useState(getToken());
  const [loadError, setLoadError] = useState("");
  const [preview, setPreview] = useState("");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const cameraInput = useRef<HTMLInputElement>(null);
  const albumInput = useRef<HTMLInputElement>(null);

  const loadTasks = () => {
    setLoadError("");
    fetchTasks()
      .then((list) => {
        setNeedToken(false);
        setTasks(list);
        setTaskId((current) => (list.some((t) => t.id === current) ? current : list[0]?.id ?? ""));
      })
      .catch((e) => (e instanceof AuthError ? setNeedToken(true) : setLoadError(e.message)));
  };
  useEffect(loadTasks, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const task = tasks?.find((t) => t.id === taskId);

  async function onPhoto(file: File | undefined) {
    if (!file || !task) return;
    setPreview(URL.createObjectURL(file));
    setStatus({ kind: "busy" });
    try {
      const photo = await preparePhoto(file);
      setStatus({ kind: "done", result: await diagnose(task.id, photo) });
    } catch (e) {
      if (e instanceof AuthError) setNeedToken(true);
      setStatus({ kind: "error", message: e instanceof Error ? e.message : "Diagnosis failed. Please try again." });
    }
  }

  if (needToken) {
    return (
      <main className="page">
        <Header />
        <section className="card">
          <p>Please enter the access code</p>
          <input className="field" type="password" value={tokenInput} autoComplete="current-password"
                 onChange={(e) => setTokenInput(e.target.value)} />
          <button className="btn primary" onClick={() => { setToken(tokenInput.trim()); loadTasks(); }}>Continue</button>
        </section>
      </main>
    );
  }

  return (
    <main className="page">
      <Header />

      <section className="card">
        <label className="label" htmlFor="task">Cancer type</label>
        {tasks === null && !loadError && <p className="muted">Connecting to the service…</p>}
        {loadError && (
          <p className="error">{loadError} <button className="link" onClick={loadTasks}>Retry</button></p>
        )}
        {tasks?.length === 0 && <p className="muted">No models are installed on the server yet</p>}
        {tasks && tasks.length > 0 && (
          <select id="task" className="field" value={taskId}
                  onChange={(e) => { setTaskId(e.target.value); setStatus({ kind: "idle" }); setPreview(""); }}>
            {tasks.map((t) => <option key={t.id} value={t.id}>{t.name} ({t.modality})</option>)}
          </select>
        )}
        {task && <p className="muted hint">{task.input_hint}</p>}
      </section>

      <div className="actions">
        <button className="btn primary" disabled={!task || status.kind === "busy"}
                onClick={() => cameraInput.current?.click()}>Take photo</button>
        <button className="btn" disabled={!task || status.kind === "busy"}
                onClick={() => albumInput.current?.click()}>Choose from library</button>
      </div>
      {/* capture="environment" opens the iPhone's rear camera directly */}
      <input ref={cameraInput} type="file" accept="image/*" capture="environment" hidden
             onChange={(e) => { onPhoto(e.target.files?.[0]); e.target.value = ""; }} />
      <input ref={albumInput} type="file" accept="image/*" hidden
             onChange={(e) => { onPhoto(e.target.files?.[0]); e.target.value = ""; }} />

      {preview && (
        <section className="card result">
          <img className="photo" src={preview} alt="Lesion photo" />
          {status.kind === "busy" && <p className="muted center"><span className="spinner" />Analysing, please wait…</p>}
          {status.kind === "error" && <p className="error">{status.message}</p>}
          {status.kind === "done" && (
            <div className={`verdict ${status.result.level}`}>
              <p className="verdict-title">{status.result.verdict}</p>
              <p>{status.result.advice}</p>
            </div>
          )}
        </section>
      )}
    </main>
  );
}

function Header() {
  return (
    <header className="header">
      <h1>Cancer Image Diagnosis</h1>
    </header>
  );
}
