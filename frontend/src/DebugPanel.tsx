import type { Health, Profile, TurnResponse } from "./api";

const FIELDS: { key: keyof Profile; label: string }[] = [
  { key: "school_year", label: "School year" },
  { key: "specialization", label: "Specialization" },
  { key: "interests", label: "Interests" },
  { key: "magic_orientation", label: "Orientation" },
  { key: "learning_modality", label: "Learning style" },
  { key: "weekly_free_hours", label: "Free hours / week" },
];

const PHASES = ["consent", "collecting", "confirming", "done"] as const;

/** Live view of what the engine has understood so far. */
export function DebugPanel({ turn, health }: { turn: TurnResponse | null; health: Health | null }) {
  return (
    <aside className="debug">
      <h2>What the advisor knows</h2>

      <ol className="phases">
        {PHASES.map((phase) => (
          <li key={phase} className={turn?.phase === phase ? "current" : ""}>
            {phase}
          </li>
        ))}
      </ol>

      <table className="profile">
        <tbody>
          {FIELDS.map(({ key, label }) => {
            const value = turn?.profile[key];
            const error = turn?.field_errors[key];
            return (
              <tr key={key}>
                <th scope="row">{label}</th>
                <td>
                  {value ?? <span className="muted">—</span>}
                  {error && <div className="field-error">{error}</div>}
                </td>
              </tr>
            );
          })}
          <tr>
            <th scope="row">Language</th>
            <td>{turn?.language ?? <span className="muted">—</span>}</td>
          </tr>
        </tbody>
      </table>

      {turn && turn.recommendations.length > 0 && (
        <>
          <h3>Recommendations</h3>
          <ul className="recommendations">
            {turn.recommendations.map((rec) => (
              <li key={rec.name}>
                <strong>{rec.name}</strong>
                <span>{rec.reason}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      {turn?.summary && (
        <>
          <h3>Summary for staff</h3>
          <ul className="summary">
            {turn.summary.map((point, index) => (
              <li key={index}>{point}</li>
            ))}
          </ul>
        </>
      )}

      <p className="muted small status">
        {health
          ? `Server ok · speech-to-text ${health.stt_provider ?? "off"} · text-to-speech ${health.text_to_speech ? "on" : "off"}`
          : "Server not reached"}
        {turn && (
          <>
            <br />
            Conversation {turn.conversation_id}
          </>
        )}
      </p>
    </aside>
  );
}
