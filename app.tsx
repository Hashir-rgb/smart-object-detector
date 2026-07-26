import { useState, useRef, useEffect, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import Sidebar, { type SourceMode } from '@/components/Sidebar';

type MediaSource = HTMLVideoElement | HTMLImageElement;

function sourceWidth(s: MediaSource): number {
  return (s as HTMLImageElement).naturalWidth || (s as HTMLVideoElement).videoWidth || s.clientWidth;
}

function sourceHeight(s: MediaSource): number {
  return (s as HTMLImageElement).naturalHeight || (s as HTMLVideoElement).videoHeight || s.clientHeight;
}
import ViewerPanel from '@/components/ViewerPanel';
import StatsPanel from '@/components/StatsPanel';
import {
  loadModel,
  detect,
  drawDetections,
  buildDetectionEvents,
  detectionsToCSV,
  downloadBlob,
  type DetectionEvent,
} from '@/lib/detector';

const MAX_LOG_EVENTS = 200;

function App() {
  const [mode, setMode] = useState<SourceMode>('idle');
  const [detecting, setDetecting] = useState(false);
  const [modelLoading, setModelLoading] = useState(true);
  const [threshold, setThreshold] = useState(0.5);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [total, setTotal] = useState(0);
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [fps, setFps] = useState(0);

  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const detectingRef = useRef(false);
  const thresholdRef = useRef(threshold);
  const modeRef = useRef<SourceMode>('idle');
  const lastFrameRef = useRef<number>(0);
  const fpsBufRef = useRef<number[]>([]);

  useEffect(() => {
    thresholdRef.current = threshold;
  }, [threshold]);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    detectingRef.current = detecting;
  }, [detecting]);

  // Pre-load the model on mount
  useEffect(() => {
    let cancelled = false;
    setModelLoading(true);
    loadModel()
      .then((m) => {
        if (cancelled) return;
        modelRef.current = m;
        setModelLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setErrorMsg(`Failed to load detection model: ${err.message}`);
        setModelLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const updateStats = useCallback((dets: cocoSsd.DetectedObject[], source: string) => {
    const thr = thresholdRef.current;
    const filtered = dets.filter((d) => d.score >= thr);
    const nextCounts: Record<string, number> = {};
    for (const d of filtered) {
      nextCounts[d.class] = (nextCounts[d.class] || 0) + 1;
    }
    setCounts(nextCounts);
    setTotal(filtered.length);

    // Log only when there are new detections to avoid spamming on static images
    if (filtered.length > 0) {
      const newEvents = buildDetectionEvents(filtered, thr, source);
      setEvents((prev) => [...newEvents.reverse(), ...prev].slice(0, MAX_LOG_EVENTS));
    }
  }, []);

  const renderFrame = useCallback(
    async (source: HTMLVideoElement | HTMLImageElement, sourceName: string) => {
      if (!detectingRef.current || !modelRef.current || !canvasRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Match canvas to the source's natural pixel dimensions so detection
      // coordinates (which are in natural pixels) line up correctly.
      // CSS scales both the media element and canvas to the same display size.
      const w = sourceWidth(source);
      const h = sourceHeight(source);
      if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
        canvas.width = w;
        canvas.height = h;
      }

      try {
        const dets = await detect(modelRef.current, source);
        drawDetections(ctx, dets, thresholdRef.current);
        updateStats(dets, sourceName);

        // FPS calc
        const now = performance.now();
        if (lastFrameRef.current > 0) {
          const delta = now - lastFrameRef.current;
          fpsBufRef.current.push(1000 / delta);
          if (fpsBufRef.current.length > 15) fpsBufRef.current.shift();
          const avg = fpsBufRef.current.reduce((a, b) => a + b, 0) / fpsBufRef.current.length;
          setFps(avg);
        }
        lastFrameRef.current = now;
      } catch {
        // detection can fail transiently during source loading; skip silently
      }

      if (detectingRef.current) {
        rafRef.current = requestAnimationFrame(() => renderFrame(source, sourceName));
      }
    },
    [updateStats]
  );

  const stopAll = useCallback(() => {
    detectingRef.current = false;
    setDetecting(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = '';
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setCounts({});
    setTotal(0);
    setFps(0);
    fpsBufRef.current = [];
    lastFrameRef.current = 0;
  }, []);

  const startDetection = useCallback(
    (source: HTMLVideoElement | HTMLImageElement, sourceName: string) => {
      if (!modelRef.current) {
        setErrorMsg('Model is still loading. Please wait a moment.');
        return;
      }
      stopAll();
      setMode(modeRef.current);
      setErrorMsg(null);
      detectingRef.current = true;
      setDetecting(true);
      lastFrameRef.current = 0;
      fpsBufRef.current = [];
      rafRef.current = requestAnimationFrame(() => renderFrame(source, sourceName));
    },
    [renderFrame, stopAll]
  );

  const handleUploadImage = useCallback(
    (file: File) => {
      if (!imageRef.current) return;
      stopAll();
      setMode('image');
      modeRef.current = 'image';
      const url = URL.createObjectURL(file);
      imageRef.current.onload = () => {
        URL.revokeObjectURL(url);
        if (imageRef.current) startDetection(imageRef.current, 'image');
      };
      imageRef.current.onerror = () => setErrorMsg('Could not load that image file.');
      imageRef.current.src = url;
    },
    [startDetection, stopAll]
  );

  const handleUploadVideo = useCallback(
    (file: File) => {
      if (!videoRef.current) return;
      stopAll();
      setMode('video');
      modeRef.current = 'video';
      const url = URL.createObjectURL(file);
      videoRef.current.onloadeddata = () => {
        videoRef.current?.play().catch(() => {});
        if (videoRef.current) startDetection(videoRef.current, 'video');
      };
      videoRef.current.onerror = () => setErrorMsg('Could not load that video file.');
      videoRef.current.src = url;
    },
    [startDetection, stopAll]
  );

  const handleStartWebcam = useCallback(async () => {
    if (!videoRef.current) return;
    stopAll();
    setMode('webcam');
    modeRef.current = 'webcam';
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play().catch(() => {});
        if (videoRef.current) startDetection(videoRef.current, 'webcam');
      };
    } catch (err) {
      setErrorMsg(
        'Could not access the webcam. ' +
          (err instanceof Error ? err.message : 'Please grant camera permission and try again.')
      );
      setMode('idle');
      modeRef.current = 'idle';
    }
  }, [startDetection, stopAll]);

  const handleStop = useCallback(() => {
    stopAll();
    setMode('idle');
    modeRef.current = 'idle';
  }, [stopAll]);

  const handleClearLog = useCallback(() => setEvents([]), []);

  const handleSnapshot = useCallback(() => {
    const source: MediaSource | null =
      modeRef.current === 'image' ? imageRef.current : modeRef.current !== 'idle' ? videoRef.current : null;
    if (!source) return;
    const w = sourceWidth(source);
    const h = sourceHeight(source);
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    const ctx = tmp.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(source, 0, 0, w, h);
    if (canvasRef.current) {
      ctx.drawImage(canvasRef.current, 0, 0, w, h);
    }
    tmp.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `snapshot-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 'image/png');
  }, []);

  const handleExportCSV = useCallback(() => {
    const csv = detectionsToCSV(events);
    downloadBlob(csv, `detections-${Date.now()}.csv`, 'text/csv');
  }, [events]);

  const handleExportJSON = useCallback(() => {
    downloadBlob(JSON.stringify(events, null, 2), `detections-${Date.now()}.json`, 'application/json');
  }, [events]);

  useEffect(() => {
    return () => stopAll();
  }, [stopAll]);

  return (
    <div className="h-screen w-screen flex bg-slate-950 overflow-hidden">
      <Sidebar
        mode={mode}
        detecting={detecting}
        modelLoading={modelLoading}
        threshold={threshold}
        onUploadImage={handleUploadImage}
        onUploadVideo={handleUploadVideo}
        onStartWebcam={handleStartWebcam}
        onStop={handleStop}
        onThresholdChange={setThreshold}
        onExportCSV={handleExportCSV}
        onExportJSON={handleExportJSON}
        onSnapshot={handleSnapshot}
        eventCount={events.length}
      />

      <main className="flex-1 flex flex-col min-w-0 ml-72 mr-80">
        <ViewerPanel
          mode={mode}
          detecting={detecting}
          modelLoading={modelLoading}
          errorMsg={errorMsg}
          fps={fps}
          videoRef={videoRef}
          imageRef={imageRef}
          canvasRef={canvasRef}
        />
      </main>

      <StatsPanel counts={counts} total={total} events={events} onClear={handleClearLog} />
    </div>
  );
}

export default App;
