import { useState } from "react"
import { toast } from "sonner"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Field,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api } from "@/lib/api"
import { useAuth } from "@/lib/auth"

export function AuthPage() {
  const { login } = useAuth()
  const [loginUser, setLoginUser] = useState("")
  const [loginPass, setLoginPass] = useState("")
  const [loginError, setLoginError] = useState<string | null>(null)
  const [loginLoading, setLoginLoading] = useState(false)

  const [fullName, setFullName] = useState("")
  const [email, setEmail] = useState("")
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [registerError, setRegisterError] = useState<string | null>(null)
  const [registerLoading, setRegisterLoading] = useState(false)

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    setLoginError(null)
    if (!loginUser || !loginPass) {
      setLoginError("Please fill in all fields.")
      return
    }
    setLoginLoading(true)
    try {
      await login(loginUser, loginPass)
      toast.success("Welcome back!")
    } catch (err) {
      setLoginError(err instanceof Error ? err.message : "Login failed")
    } finally {
      setLoginLoading(false)
    }
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault()
    setRegisterError(null)

    if (!fullName || !email || !username || !password || !confirmPassword) {
      setRegisterError("Please fill in all fields.")
      return
    }
    if (username.length < 3) {
      setRegisterError("Username must be at least 3 characters long.")
      return
    }
    if (password.length < 6) {
      setRegisterError("Password must be at least 6 characters long.")
      return
    }
    if (password !== confirmPassword) {
      setRegisterError("Passwords do not match.")
      return
    }
    if (!email.includes("@") || !email.includes(".")) {
      setRegisterError("Please enter a valid email address.")
      return
    }

    setRegisterLoading(true)
    try {
      const res = await api.register({
        full_name: fullName,
        email,
        username,
        password,
        confirm_password: confirmPassword,
      })
      toast.success(res.message || "Account created successfully!")
      setFullName("")
      setEmail("")
      setUsername("")
      setPassword("")
      setConfirmPassword("")
    } catch (err) {
      setRegisterError(err instanceof Error ? err.message : "Registration failed")
    } finally {
      setRegisterLoading(false)
    }
  }

  return (
    <div className="flex min-h-svh items-center justify-center bg-background p-6">
      <div className="flex w-full max-w-md flex-col gap-6">
        <div className="flex flex-col gap-2 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            PDF Summarizer
          </h1>
          <p className="text-sm text-muted-foreground">
            Welcome! Please login or register to access the PDF Summarizer tool.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Account</CardTitle>
            <CardDescription>
              Only approved company employees can access this tool. New
              registrations require admin approval.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="login">
              <TabsList className="w-full">
                <TabsTrigger value="login">Login</TabsTrigger>
                <TabsTrigger value="register">Register</TabsTrigger>
              </TabsList>

              <TabsContent value="login" className="pt-4">
                <form onSubmit={handleLogin}>
                  <FieldGroup>
                    <Field>
                      <FieldLabel htmlFor="login-username">Username</FieldLabel>
                      <Input
                        id="login-username"
                        value={loginUser}
                        onChange={(e) => setLoginUser(e.target.value)}
                        placeholder="Enter your username"
                        autoComplete="username"
                      />
                    </Field>
                    <Field>
                      <FieldLabel htmlFor="login-password">Password</FieldLabel>
                      <Input
                        id="login-password"
                        type="password"
                        value={loginPass}
                        onChange={(e) => setLoginPass(e.target.value)}
                        placeholder="Enter your password"
                        autoComplete="current-password"
                      />
                    </Field>
                    {loginError ? (
                      <Alert variant="destructive">
                        <AlertDescription>{loginError}</AlertDescription>
                      </Alert>
                    ) : null}
                    <Button type="submit" className="w-full" disabled={loginLoading}>
                      {loginLoading ? <Spinner data-icon="inline-start" /> : null}
                      Login
                    </Button>
                  </FieldGroup>
                </form>
              </TabsContent>

              <TabsContent value="register" className="pt-4">
                <form onSubmit={handleRegister}>
                  <FieldGroup>
                    <Field>
                      <FieldLabel htmlFor="reg-name">Full Name</FieldLabel>
                      <Input
                        id="reg-name"
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        placeholder="Enter your full name"
                      />
                    </Field>
                    <Field>
                      <FieldLabel htmlFor="reg-email">Email</FieldLabel>
                      <Input
                        id="reg-email"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="Enter your company email"
                      />
                    </Field>
                    <Field>
                      <FieldLabel htmlFor="reg-username">Username</FieldLabel>
                      <Input
                        id="reg-username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="Choose a username"
                      />
                    </Field>
                    <Field>
                      <FieldLabel htmlFor="reg-password">Password</FieldLabel>
                      <Input
                        id="reg-password"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Choose a password (min 6 characters)"
                      />
                    </Field>
                    <Field>
                      <FieldLabel htmlFor="reg-confirm">Confirm Password</FieldLabel>
                      <Input
                        id="reg-confirm"
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="Re-enter your password"
                      />
                    </Field>
                    {registerError ? (
                      <Alert variant="destructive">
                        <AlertDescription>{registerError}</AlertDescription>
                      </Alert>
                    ) : null}
                    <Button
                      type="submit"
                      className="w-full"
                      disabled={registerLoading}
                    >
                      {registerLoading ? (
                        <Spinner data-icon="inline-start" />
                      ) : null}
                      Register
                    </Button>
                  </FieldGroup>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
