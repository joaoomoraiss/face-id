import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router";
import { registerUser } from "../api";
import { CameraCapture } from "../CameraCapture";

export default function RegisterPage() {
  const navigate = useNavigate();
  const cameraRef = useRef(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const image = await cameraRef.current.capture();
      await registerUser(name, email, image);
      navigate("/login", {
        state: { message: "Cadastro feito. Entre com a mesma foto." },
      });
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
              <h1 className="h3 mb-3">Criar conta</h1>
              {error ? <div className="alert alert-danger">{error}</div> : null}
              <form onSubmit={onSubmit}>
                <div className="mb-3">
                  <label className="form-label" htmlFor="name">Nome</label>
                  <input
                    id="name"
                    className="form-control"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    required
                  />
                </div>
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
                  {loading ? "Cadastrando..." : "Cadastrar rosto"}
                </button>
              </form>
              <p className="mt-3 mb-0">
                <Link to="/login">Já tenho conta</Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
