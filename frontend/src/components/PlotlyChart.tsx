import { lazy, Suspense } from "react";
import type { Data, Layout } from "plotly.js";

const Plot = lazy(() => import("react-plotly.js"));

interface Props {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
}

export function PlotlyChart({ data, layout, height = 320 }: Props) {
  const mergedLayout: Partial<Layout> = {
    autosize: true,
    margin: { t: 32, r: 16, b: 40, l: 48 },
    legend: { orientation: "h" },
    ...layout,
  };

  return (
    <Suspense fallback={<div className="h-80 grid place-items-center text-slate-400">Loading chart…</div>}>
      <Plot
        data={data}
        layout={mergedLayout}
        useResizeHandler
        style={{ width: "100%", height }}
        config={{ displaylogo: false, responsive: true }}
      />
    </Suspense>
  );
}
