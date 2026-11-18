export async function fetchJson<T = Record<string, unknown>>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    const body = await response.text()
    let detail = body
    if (body) {
      try {
        const parsed: unknown = JSON.parse(body)
        if (parsed && typeof parsed === 'object' && 'detail' in parsed) {
          const apiDetail = parsed.detail
          if (typeof apiDetail === 'string') detail = apiDetail
        }
      } catch {
        detail = body
      }
    }
    throw new Error(`Request failed for ${url}: ${response.status}${detail ? ` — ${detail}` : ''}`)
  }
  return (await response.json()) as T
}
