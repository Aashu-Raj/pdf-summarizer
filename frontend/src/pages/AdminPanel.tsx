import { useCallback, useEffect, useState } from "react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import { api, type AdminUser } from "@/lib/api"
import { useAuth } from "@/lib/auth"

export function AdminPanel() {
  const { user } = useAuth()
  const [loading, setLoading] = useState(true)
  const [users, setUsers] = useState<AdminUser[]>([])
  const [pending, setPending] = useState<AdminUser[]>([])
  const [stats, setStats] = useState({ approved: 0, pending: 0, rejected: 0 })
  const [busy, setBusy] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.adminUsers()
      setUsers(data.users)
      setPending(data.pending)
      setStats(data.stats)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to load users")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  async function run(key: string, fn: () => Promise<unknown>) {
    setBusy(key)
    try {
      await fn()
      await load()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Action failed")
    } finally {
      setBusy(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Spinner /> Loading admin panel…
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-base font-semibold">Admin Panel</h2>
        <p className="text-sm text-muted-foreground">Manage users and approvals</p>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-medium">
          Pending Approvals ({pending.length})
        </h3>
        {pending.length === 0 ? (
          <p className="text-sm text-muted-foreground">No pending approvals</p>
        ) : (
          <div className="flex flex-col gap-3">
            {pending.map((u) => (
              <Card key={u.username}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">
                    {u.full_name}{" "}
                    <span className="font-normal text-muted-foreground">
                      ({u.username})
                    </span>
                  </CardTitle>
                  <CardDescription>
                    {u.email}
                    {u.created_at ? ` · Registered: ${u.created_at}` : null}
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex gap-2">
                  <Button
                    size="sm"
                    disabled={busy !== null}
                    onClick={() =>
                      void run(`approve-${u.username}`, async () => {
                        await api.updateUserStatus(u.username, "approved")
                        toast.success(`Approved ${u.username}`)
                      })
                    }
                  >
                    {busy === `approve-${u.username}` ? (
                      <Spinner data-icon="inline-start" />
                    ) : null}
                    Approve
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    disabled={busy !== null}
                    onClick={() =>
                      void run(`reject-${u.username}`, async () => {
                        await api.updateUserStatus(u.username, "rejected")
                        toast.success(`Rejected ${u.username}`)
                      })
                    }
                  >
                    Reject
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      <Separator />

      <div>
        <h3 className="mb-2 text-sm font-medium">All Users</h3>
        <div className="mb-3 flex gap-2">
          <Badge variant="secondary">Approved {stats.approved}</Badge>
          <Badge variant="secondary">Pending {stats.pending}</Badge>
          <Badge variant="secondary">Rejected {stats.rejected}</Badge>
        </div>

        <div className="flex flex-col gap-3">
          {users.map((u) => {
            const isSelf = u.username === user?.username
            const newRole = u.role === "admin" ? "user" : "admin"
            return (
              <Card key={u.username}>
                <CardHeader className="pb-2">
                  <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
                    {u.full_name}
                    <Badge variant="outline">{u.role}</Badge>
                    <Badge
                      variant={
                        u.status === "approved"
                          ? "default"
                          : u.status === "pending"
                            ? "secondary"
                            : "destructive"
                      }
                    >
                      {u.status}
                    </Badge>
                  </CardTitle>
                  <CardDescription>
                    {u.email}
                    {u.created_at ? ` · ${u.created_at}` : null}
                  </CardDescription>
                </CardHeader>
                {!isSelf ? (
                  <CardContent className="flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={busy !== null}
                      onClick={() =>
                        void run(`role-${u.username}`, async () => {
                          await api.updateUserRole(u.username, newRole)
                          toast.success(`Changed ${u.username} to ${newRole}`)
                        })
                      }
                    >
                      Make {newRole}
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      disabled={busy !== null}
                      onClick={() =>
                        void run(`delete-${u.username}`, async () => {
                          await api.deleteUser(u.username)
                          toast.success(`Deleted ${u.username}`)
                        })
                      }
                    >
                      Delete
                    </Button>
                  </CardContent>
                ) : (
                  <CardContent>
                    <p className="text-xs text-muted-foreground">This is your account</p>
                  </CardContent>
                )}
              </Card>
            )
          })}
        </div>
      </div>
    </div>
  )
}
