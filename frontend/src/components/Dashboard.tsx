import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  getDatasetSummary,
  getHealthIndicator,
  getJobStatus,
  getMetrics,
  getPredictions,
  runPipeline,
} from "../api/client";
import type { JobStatus } from "../api/types";
import { PlotlyChart } from "./PlotlyChart";

const TERMINAL = new Set(["succeeded", "failed"]);

export function Dashboard() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [useLstm, setUseLstm] = useState(true);

  const startRun = useMutation({
    mutationFn: () => runPipeline({ use_lstm: useLstm }),
    onSuccess: (data) => setJobId(data.job_id),
  });

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && TERMINAL.has(status) ? false : 1500;
    },
  });

  const isReady = jobQuery.data?.status === "succeeded";

  return (
    <div className="mx-auto max-w-6xl p-6 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Wind Turbine RUL</h1>
          <p className="text-sm text-slate-500">
            Health indicator, RUL predictions, and model metrics for the high-speed bearing dataset.
          </p>
        </div>
        <RunControls
          useLstm={useLstm}
          setUseLstm={setUseLstm}
          onRun={() => startRun.mutate()}
          running={startRun.isPending || (jobQuery.data?.status === "running")}
        />
      </header>

      {startRun.isError && (
        <Banner tone="error">Failed to start pipeline. Check the API is running at :8000.</Banner>
      )}

      {jobId && jobQuery.data && <JobProgress status={jobQuery.data} />}

      {isReady && jobId && <Results jobId={jobId} />}
    </div>
  );
}

function RunControls(props: {
  useLstm: boolean;
  setUseLstm: (v: boolean) => void;
  onRun: () => void;
  running: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={props.useLstm}
          onChange={(e) => props.setUseLstm(e.target.checked)}
        />
        Use LSTM
      </label>
      <button
        className="rounded bg-slate-900 text-white px-4 py-2 text-sm disabled:opacity-50"
        disabled={props.running}
        onClick={props.onRun}
      >
        {props.running ? "Running…" : "Run pipeline"}
      </button>
    </div>
  );
}

function Banner({ children, tone }: { children: React.ReactNode; tone: "info" | "error" }) {
  const color = tone === "error" ? "bg-rose-50 text-rose-700" : "bg-slate-100 text-slate-700";
  return <div className={`rounded p-3 text-sm ${color}`}>{children}</div>;
}

function JobProgress({ status }: { status: JobStatus }) {
  const pct = Math.round((status.progress ?? 0) * 100);
  return (
    <section className="rounded border bg-white p-4 space-y-2">
      <div className="flex items-baseline justify-between">
        <div className="font-medium">Job {status.job_id.slice(0, 8)}…</div>
        <div className="text-sm text-slate-500">
          {status.status} — step: {status.step ?? "—"}
        </div>
      </div>
      <div className="h-2 rounded bg-slate-100 overflow-hidden">
        <div className="h-full bg-slate-900" style={{ width: `${pct}%` }} />
      </div>
      {status.error && <Banner tone="error">{status.error}</Banner>}
    </section>
  );
}

function Results({ jobId }: { jobId: string }) {
  const summary = useQuery({ queryKey: ["summary", jobId], queryFn: () => getDatasetSummary(jobId) });
  const hi = useQuery({ queryKey: ["hi", jobId], queryFn: () => getHealthIndicator(jobId) });
  const preds = useQuery({ queryKey: ["preds", jobId], queryFn: () => getPredictions(jobId) });
  const metrics = useQuery({ queryKey: ["metrics", jobId], queryFn: () => getMetrics(jobId) });

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <section className="rounded border bg-white p-4 md:col-span-2">
        <h2 className="font-medium mb-2">Dataset</h2>
        {summary.data ? (
          <dl className="grid grid-cols-4 gap-4 text-sm">
            <Stat label="samples" value={summary.data.n_samples} />
            <Stat label="start" value={summary.data.date_start} />
            <Stat label="end" value={summary.data.date_end} />
            <Stat label="duration (d)" value={summary.data.duration_days} />
          </dl>
        ) : (
          <Skeleton />
        )}
      </section>

      <section className="rounded border bg-white p-4 md:col-span-2">
        <h2 className="font-medium mb-2">Health indicator</h2>
        {hi.data ? (
          <PlotlyChart
            data={[
              {
                x: hi.data.timestamps,
                y: hi.data.values,
                type: "scatter",
                mode: "lines+markers",
                name: "HI",
                line: { color: "#0f172a" },
              },
            ]}
            layout={{
              yaxis: { title: { text: "PCA component" } },
              shapes:
                hi.data.threshold !== null
                  ? [
                      {
                        type: "line",
                        xref: "paper",
                        x0: 0,
                        x1: 1,
                        y0: hi.data.threshold,
                        y1: hi.data.threshold,
                        line: { color: "#ef4444", dash: "dash" },
                      },
                    ]
                  : [],
            }}
          />
        ) : (
          <Skeleton />
        )}
      </section>

      <section className="rounded border bg-white p-4 md:col-span-2">
        <h2 className="font-medium mb-2">RUL predictions</h2>
        {preds.data ? (
          <PlotlyChart
            data={Object.entries(preds.data.models).flatMap(([name, m]) => [
              {
                x: m.targets.map((_, i) => i + m.sequence_offset),
                y: m.targets,
                type: "scatter",
                mode: "lines",
                name: `${name} · target`,
                line: { color: "#64748b", dash: "dot" },
              },
              {
                x: m.predictions.map((_, i) => i + m.sequence_offset),
                y: m.predictions,
                type: "scatter",
                mode: "lines+markers",
                name: `${name} · pred`,
              },
            ])}
            layout={{
              yaxis: { title: { text: "RUL (days)" } },
              xaxis: { title: { text: "sample" } },
            }}
            height={360}
          />
        ) : (
          <Skeleton />
        )}
      </section>

      <section className="rounded border bg-white p-4 md:col-span-2">
        <h2 className="font-medium mb-2">
          Metrics {metrics.data?.best_model && <span className="text-sm text-emerald-600">· best: {metrics.data.best_model}</span>}
        </h2>
        {metrics.data ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="py-1">Model</th>
                <th>MAE (days)</th>
                <th>RMSE (days)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics.data.metrics).map(([name, m]) => (
                <tr key={name} className="border-t">
                  <td className="py-1">{name}</td>
                  <td>{m.mae_days.toFixed(3)}</td>
                  <td>{m.rmse_days.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Skeleton />
        )}
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}

function Skeleton() {
  return <div className="h-24 rounded bg-slate-100 animate-pulse" />;
}
