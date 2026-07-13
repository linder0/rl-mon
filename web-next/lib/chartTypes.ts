/** A single line series with an optional symmetric band (e.g. mean +/- std). */
export interface Series {
  name: string;
  color: string;
  x: number[];
  y: number[];
  band?: { lo: number[]; hi: number[] };
}

/** The full set of dashboard charts, emitted whenever the live/compared runs
 * change. Each field is an overlay of one or more series. */
export interface ChartPayload {
  evalOverlay: Series[];
  epLenOverlay: Series[];
  train: Series[];
  approxKl: Series[];
  explainedVar: Series[];
  entropy: Series[];
}
