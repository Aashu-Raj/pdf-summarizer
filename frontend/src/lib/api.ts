const API_BASE = import.meta.env.VITE_API_URL ?? "/api"

export type User = {
  username: string
  role: string
  full_name: string
  email?: string
  status?: string
}

export type DocFile = {
  name: string
  size_mb: number
}

export type SourceDoc = {
  index: number
  preview: string
  source: string
}

export type AdminUser = {
  id: number
  username: string
  email: string
  full_name: string
  role: string
  status: string
  created_at: string | null
}

type ApiError = {
  detail?: string | { msg: string }[]
}

function getToken(): string | null {
  return localStorage.getItem("access_token")
}

export function setToken(token: string | null) {
  if (token) {
    localStorage.setItem("access_token", token)
  } else {
    localStorage.removeItem("access_token")
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as ApiError
    if (typeof data.detail === "string") return data.detail
    if (Array.isArray(data.detail)) {
      return data.detail.map((d) => d.msg).join(", ")
    }
  } catch {
    /* ignore */
  }
  return res.statusText || "Request failed"
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  auth = true
): Promise<T> {
  const headers = new Headers(options.headers)
  if (auth) {
    const token = getToken()
    if (token) headers.set("Authorization", `Bearer ${token}`)
  }
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })
  if (!res.ok) {
    throw new Error(await parseError(res))
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  login(username: string, password: string) {
    return request<{ access_token: string; user: User }>(
      "/auth/login",
      { method: "POST", body: JSON.stringify({ username, password }) },
      false
    )
  },
  register(body: {
    full_name: string
    email: string
    username: string
    password: string
    confirm_password: string
  }) {
    return request<{ message: string }>(
      "/auth/register",
      { method: "POST", body: JSON.stringify(body) },
      false
    )
  },
  me() {
    return request<User>("/auth/me")
  },
  status() {
    return request<{
      openai_connected: boolean
      database_ready: boolean
      doc_count: number
      files: DocFile[]
    }>("/status")
  },
  listDocuments() {
    return request<{
      files: DocFile[]
      database_ready: boolean
      doc_count: number
    }>("/documents")
  },
  processDocuments(files: File[]) {
    const form = new FormData()
    for (const file of files) form.append("files", file)
    return request<{
      message: string
      database_ready: boolean
      doc_count: number
      files: DocFile[]
    }>("/documents/process", { method: "POST", body: form })
  },
  clearDocuments() {
    return request<{ message: string }>("/documents", { method: "DELETE" })
  },
  ask(question: string) {
    return request<{
      question: string
      answer: string
      sources: SourceDoc[]
    }>("/ask", { method: "POST", body: JSON.stringify({ question }) })
  },
  adminUsers() {
    return request<{
      users: AdminUser[]
      pending: AdminUser[]
      stats: { approved: number; pending: number; rejected: number }
    }>("/admin/users")
  },
  updateUserStatus(username: string, status: "approved" | "pending" | "rejected") {
    return request<{ message: string }>(`/admin/users/${username}/status`, {
      method: "PATCH",
      body: JSON.stringify({ status }),
    })
  },
  updateUserRole(username: string, role: "admin" | "user") {
    return request<{ message: string }>(`/admin/users/${username}/role`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    })
  },
  deleteUser(username: string) {
    return request<{ message: string }>(`/admin/users/${username}`, {
      method: "DELETE",
    })
  },
}
