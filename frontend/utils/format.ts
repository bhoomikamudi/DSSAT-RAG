/**
 * Display formatting for calculated statistics.
 *
 * Measured quantities (yield, biomass, rainfall, temperature and their
 * averages, totals, minima, maxima, standard deviations and intercepts) are
 * shown as whole numbers with thousands separators: 4502.36 -> "4,502".
 * Values are only rounded for display; the data keeps full precision.
 *
 * Dimensionless statistics and rates (r, R², slopes) keep their decimals,
 * because rounding them to whole numbers would erase their meaning.
 */

const WHOLE = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })

/** A measured statistic, rounded half away from zero: 4502.36 -> "4,502". */
export const formatStat = (value: number | null | undefined): string =>
  typeof value === 'number' && Number.isFinite(value) ? WHOLE.format(value) : '—'

/** A coefficient such as r or R², with a fixed number of decimals. */
export const formatCoefficient = (value: number | null | undefined, digits = 3): string =>
  typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : '—'
