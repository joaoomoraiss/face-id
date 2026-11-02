import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { fetchMe } from "../api";
import { clearToken } from "../auth";

export default function HomePage() {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    fetchMe()
      .then((data) => {
        if (active) {
          setUser(data);
        }
      })
      .catch((err) => {
        if (!active) {
          return;
        }
        if (err.status === 401) {
          navigate("/login");
          return;
        }
        setError(err.message);
      });
    return () => {
      active = false;
    };
  }, [navigate]);

  function logout() {
    clearToken();
    navigate("/login");
  }

  return (
    <main className="container py-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1 className="h3 mb-0">FaceCheck</h1>
        <button className="btn btn-outline-secondary" type="button" onClick={logout}>
          Sair
        </button>
      </div>
      {error ? <div className="alert alert-danger">{error}</div> : null}
      {user ? (
        <div className="card">
          <div className="card-body">
            <p className="mb-1"><strong>Nome:</strong> {user.name}</p>
            <p className="mb-1"><strong>E-mail:</strong> {user.email}</p>
            <p className="mb-0"><strong>Perfil:</strong> {user.role}</p>
          </div>
        </div>
      ) : (
        <p>Carregando perfil...</p>
      )}
    </main>
  );
}
