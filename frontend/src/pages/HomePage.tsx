import { useCallback, useEffect, useState } from "react"
import {
  FileTextIcon,
  LogOutIcon,
  RefreshCwIcon,
  SearchIcon,
  Trash2Icon,
  UploadIcon,
} from "lucide-react"
import { toast } from "sonner"

import { AdminPanel } from "@/pages/AdminPanel"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import { Textarea } from "@/components/ui/textarea"
import { api, type DocFile, type SourceDoc } from "@/lib/api"
import { useAuth } from "@/lib/auth"

export function HomePage() {
  const { user, logout } = useAuth()
  const [openaiOk, setOpenaiOk] = useState(false)
  const [databaseReady, setDatabaseReady] = useState(false)
  const [docCount, setDocCount] = useState(0)
  const [files, setFiles] = useState<DocFile[]>([])
  const [selected, setSelected] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [question, setQuestion] = useState("")
  const [asking, setAsking] = useState(false)
  const [answer, setAnswer] = useState<string | null>(null)
  const [sources, setSources] = useState<SourceDoc[]>([])
  const [showAbout, setShowAbout] = useState(false)

  const refreshStatus = useCallback(async () => {
    try {
      const status = await api.status()
      setOpenaiOk(status.openai_connected)
      setDatabaseReady(status.database_ready)
      setDocCount(status.doc_count)
      setFiles(status.files)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to load status")
    }
  }, [])

  useEffect(() => {
    void refreshStatus()
  }, [refreshStatus])

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const list = Array.from(e.target.files ?? []).filter((f) =>
      f.name.toLowerCase().endsWith(".pdf")
    )
    setSelected(list)
  }

  async function processDocuments() {
    if (selected.length === 0) return
    setProcessing(true)
    try {
      const res = await api.processDocuments(selected)
      toast.success(res.message)
      setSelected([])
      setDatabaseReady(res.database_ready)
      setDocCount(res.doc_count)
      setFiles(res.files)
      await refreshStatus()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Processing failed")
    } finally {
      setProcessing(false)
    }
  }

  async function clearAll() {
    setClearing(true)
    try {
      const res = await api.clearDocuments()
      toast.success(res.message)
      setDatabaseReady(false)
      setDocCount(0)
      setFiles([])
      setAnswer(null)
      setSources([])
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Clear failed")
    } finally {
      setClearing(false)
    }
  }

  async function search() {
    const q = question.trim()
    if (!q) {
      toast.warning("Please enter a question to search.")
      return
    }
    setAsking(true)
    setAnswer(null)
    setSources([])
    try {
      const res = await api.ask(q)
      setAnswer(res.answer)
      setSources(res.sources)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Search failed")
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="min-h-svh bg-background">
      <div className="mx-auto grid max-w-6xl gap-6 p-6 lg:grid-cols-[1fr_280px]">
        <main className="flex min-w-0 flex-col gap-6">
          <div className="flex flex-col gap-2">
            <h1 className="text-2xl font-semibold tracking-tight">
              Search your PDF with OpenAI
            </h1>
            {openaiOk ? (
              <Alert>
                <AlertTitle>OpenAI API Key loaded</AlertTitle>
                <AlertDescription>
                  Connected and ready to process documents.
                </AlertDescription>
              </Alert>
            ) : (
              <Alert variant="destructive">
                <AlertTitle>OpenAI API Key not found</AlertTitle>
                <AlertDescription>
                  Ensure your server <code>.env</code> contains{" "}
                  <code>OPENAI_API_KEY</code>.
                </AlertDescription>
              </Alert>
            )}
          </div>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2">
              <div>
                <CardTitle>About the App</CardTitle>
                <CardDescription>
                  Generative AI Q&amp;A over your PDF documents
                </CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowAbout((v) => !v)}
              >
                {showAbout ? "Hide" : "Show"}
              </Button>
            </CardHeader>
            {showAbout ? (
              <CardContent className="text-sm text-muted-foreground">
                <ol className="list-decimal pl-5">
                  <li>Upload your PDF files</li>
                  <li>Click Process Documents to index them</li>
                  <li>Ask questions about your documents</li>
                  <li>Get AI-powered answers with source references</li>
                </ol>
              </CardContent>
            ) : null}
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Document Upload &amp; Processing</CardTitle>
              <CardDescription>Upload one or more PDF files</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <FieldGroup>
                <Field>
                  <FieldLabel htmlFor="pdf-upload">Choose PDF files</FieldLabel>
                  <InputFile id="pdf-upload" onChange={onFileChange} />
                </Field>
              </FieldGroup>

              {selected.length > 0 ? (
                <div className="flex flex-col gap-1 text-sm">
                  <p className="font-medium">Selected files</p>
                  {selected.map((f) => (
                    <p key={f.name} className="text-muted-foreground">
                      {f.name} ({(f.size / (1024 * 1024)).toFixed(2)} MB)
                    </p>
                  ))}
                </div>
              ) : null}

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  onClick={() => void processDocuments()}
                  disabled={selected.length === 0 || processing || !openaiOk}
                >
                  {processing ? (
                    <Spinner data-icon="inline-start" />
                  ) : (
                    <RefreshCwIcon data-icon="inline-start" />
                  )}
                  Process Documents
                </Button>
                {(databaseReady || files.length > 0) && (
                  <Button
                    variant="outline"
                    onClick={() => void clearAll()}
                    disabled={clearing}
                  >
                    {clearing ? (
                      <Spinner data-icon="inline-start" />
                    ) : (
                      <Trash2Icon data-icon="inline-start" />
                    )}
                    Clear All Data
                  </Button>
                )}
                {selected.length > 0 ? (
                  <Badge variant="secondary">{selected.length} file(s) ready</Badge>
                ) : (
                  <Badge variant="outline">
                    <UploadIcon data-icon="inline-start" />
                    Upload PDFs
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>

          {databaseReady ? (
            <Card>
              <CardHeader>
                <CardTitle>Ask Questions</CardTitle>
                <CardDescription>
                  Document database is ready. Ask anything about your PDFs.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                <FieldGroup>
                  <Field>
                    <FieldLabel htmlFor="question">
                      Enter your question about the documents
                    </FieldLabel>
                    <Textarea
                      id="question"
                      rows={4}
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Example: What are the main topics discussed in these documents?"
                    />
                  </Field>
                </FieldGroup>
                <Button
                  className="w-full"
                  onClick={() => void search()}
                  disabled={asking || !openaiOk}
                >
                  {asking ? (
                    <Spinner data-icon="inline-start" />
                  ) : (
                    <SearchIcon data-icon="inline-start" />
                  )}
                  Search
                </Button>

                {answer ? (
                  <div className="flex flex-col gap-3">
                    <div>
                      <h3 className="mb-1 text-sm font-medium">Answer</h3>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">
                        {answer}
                      </p>
                    </div>
                    {sources.length > 0 ? (
                      <div className="flex flex-col gap-2">
                        <h3 className="text-sm font-medium">Source Documents</h3>
                        {sources.map((s) => (
                          <Card key={s.index}>
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm">
                                Source {s.index}
                              </CardTitle>
                              <CardDescription>{s.source}</CardDescription>
                            </CardHeader>
                            <CardContent>
                              <pre className="overflow-x-auto whitespace-pre-wrap rounded-md bg-muted p-3 font-mono text-xs">
                                {s.preview}
                              </pre>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No source documents found for this query.
                      </p>
                    )}
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ) : (
            <Empty className="border">
              <EmptyHeader>
                <EmptyTitle>No searchable database yet</EmptyTitle>
                <EmptyDescription>
                  {selected.length > 0
                    ? "Files selected. Click Process Documents to create the searchable database."
                    : "Upload PDF files first to get started."}
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </main>

        <aside className="flex flex-col gap-4 lg:sticky lg:top-6 lg:self-start">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                {user?.full_name ?? "User"}
              </CardTitle>
              <CardDescription>
                Role: {user?.role ? user.role[0].toUpperCase() + user.role.slice(1) : "—"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="outline" className="w-full" onClick={logout}>
                <LogOutIcon data-icon="inline-start" />
                Logout
              </Button>
            </CardContent>
          </Card>

          {user?.role === "admin" ? (
            <Card>
              <CardContent className="pt-6">
                <AdminPanel />
              </CardContent>
            </Card>
          ) : null}

          <Card>
            <CardHeader>
              <CardTitle className="text-base">System Status</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2 text-sm">
              <StatusRow
                ok={openaiOk}
                label={openaiOk ? "OpenAI API: Connected" : "OpenAI API: Not Connected"}
              />
              <StatusRow
                ok={databaseReady}
                label={
                  databaseReady
                    ? "Vector Database: Ready"
                    : "Vector Database: Not Created"
                }
              />
              {databaseReady && docCount > 0 ? (
                <p className="flex items-center gap-2 text-muted-foreground">
                  <FileTextIcon className="size-4" />
                  Documents: {docCount} PDF(s) processed
                </p>
              ) : null}
              {files.length > 0 ? (
                <>
                  <Separator />
                  <ul className="flex flex-col gap-1 text-muted-foreground">
                    {files.map((f) => (
                      <li key={f.name}>
                        {f.name} ({f.size_mb} MB)
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Tips</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              <ul className="list-disc pl-4">
                <li>Be specific and detailed in questions</li>
                <li>Reference key terms from your documents</li>
                <li>Ensure PDFs contain selectable text</li>
                <li>Larger documents take longer to process</li>
              </ul>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  )
}

function StatusRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <p className={ok ? "text-foreground" : "text-destructive"}>
      {ok ? "✓" : "✕"} {label}
    </p>
  )
}

function InputFile({
  id,
  onChange,
}: {
  id: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
}) {
  return (
    <input
      id={id}
      type="file"
      accept=".pdf,application/pdf"
      multiple
      onChange={onChange}
      className="block w-full cursor-pointer text-sm text-muted-foreground file:mr-3 file:rounded-lg file:border-0 file:bg-primary file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-primary-foreground"
    />
  )
}
