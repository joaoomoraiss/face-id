import { useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import { loginUser } from "../api";
import { CameraCapture } from "../CameraCapture";

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const cameraRef = useRef(null);
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const notice = location.state?.message ?? "";

  async function onSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const image = await cameraRef.current.capture();
      await loginUser(email, image);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container py-5">
      <div className="row justify-content-center">
        <div className="col-md-6">
          <div className="card">
            <div className="card-body">
              <h1 className="h3 mb-3">Entrar</h1>
              {notice ? <div className="alert alert-success">{notice}</div> : null}
              {error ? <div className="alert alert-danger">{error}</div> : null}
              <form onSubmit={onSubmit}>
                <div className="mb-3">
                  <label className="form-label" htmlFor="email">E-mail</label>
                  <input
                    id="email"
                    className="form-control"
                    type="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                  />
                </div>
                <CameraCapture ref={cameraRef} />
                <button className="btn btn-primary" type="submit" disabled={loading}>
                  {loading ? "Entrando..." : "Entrar"}
                </button>
              </form>
              <p className="mt-3 mb-0">
                <Link to="/register">Criar uma conta</Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
