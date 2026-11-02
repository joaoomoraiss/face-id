import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

export const CameraCapture = forwardRef(function CameraCapture(_props, ref) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Este navegador não oferece webcam");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch {
        if (!cancelled) {
          setError("Não foi possível acessar a webcam");
        }
      }
    }

    start();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useImperativeHandle(ref, () => ({
    async capture() {
      const video = videoRef.current;
      if (!video || !video.videoWidth || !video.videoHeight) {
        throw new Error("A webcam ainda não está pronta");
      }
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Não foi possível processar a foto");
      }
      context.drawImage(video, 0, 0);
      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.92);
      });
      if (!blob) {
        throw new Error("Não foi possível processar a foto");
      }
      return new File([blob], "rosto.jpg", { type: "image/jpeg" });
    },
  }));

  return (
    <div className="mb-3">
      <p className="form-label mb-2">Foto do rosto</p>
      {error ? <div className="alert alert-danger">{error}</div> : null}
      {!error && !ready ? <p className="text-secondary">Ativando a webcam...</p> : null}
      <video
        ref={videoRef}
        className="w-100 rounded border bg-dark"
        style={{ transform: "scaleX(-1)", display: error ? "none" : "block" }}
        autoPlay
        playsInline
        muted
        onPlaying={() => setReady(true)}
      />
    </div>
  );
});
