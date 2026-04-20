/// <reference types="vite/client" />

declare module "react-plotly.js" {
  import type { Component } from "react";
  import type { Data, Layout, Config } from "plotly.js";
  export interface PlotParams {
    data: Data[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    style?: React.CSSProperties;
    useResizeHandler?: boolean;
    onInitialized?: (figure: unknown, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: unknown, graphDiv: HTMLElement) => void;
  }
  export default class Plot extends Component<PlotParams> {}
}

declare module "plotly.js-basic-dist-min" {
  const plotly: unknown;
  export default plotly;
}
