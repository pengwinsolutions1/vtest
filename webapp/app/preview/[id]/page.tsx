'use client';
// Standalone GLB preview at /preview/<garment-id>?angle=front|side|back
// No webcam, no MediaPipe — just centres the GLB at origin, places a camera
// on a sphere of radius 2.5 at the requested angle, lights it, renders. Used
// for diagnosing whether the GLB itself is correctly built independent of
// the live AR overlay logic.
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { use } from 'react';

interface Garment { id: string; name: string; glb_url?: string | null; }

export default function Preview({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const containerRef = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState<string>('loading…');

  useEffect(() => {
    let cleanup = () => {};
    (async () => {
      // Find the garment
      const list = await fetch('/api/garments').then(r => r.json());
      const g: Garment | undefined = list.garments?.find((x: Garment) => x.id === id);
      if (!g || !g.glb_url) { setInfo(`no GLB for ${id}`); return; }

      // Read ?angle= from URL
      const angle = new URLSearchParams(window.location.search).get('angle') || 'front';

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.setClearColor(0xffffff, 1);
      containerRef.current!.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      scene.add(new THREE.HemisphereLight(0xffffff, 0xbfbfbf, 1.0));
      const key = new THREE.DirectionalLight(0xffffff, 1.0);
      key.position.set(1, 1.5, 1.5); scene.add(key);

      const cam = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.01, 100);
      const R = 2.5;
      const theta = angle === 'side' ? Math.PI / 2 : angle === 'back' ? Math.PI : 0;
      cam.position.set(R * Math.sin(theta), 0.1, R * Math.cos(theta));
      cam.lookAt(0, 0, 0);

      const loader = new GLTFLoader();
      const gltf = await loader.loadAsync(g.glb_url);
      const root = gltf.scene;
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const centre = box.getCenter(new THREE.Vector3());
      root.position.sub(centre);
      // Force DoubleSide so we can see back faces if culling is the issue
      root.traverse((o: any) => {
        if (o.isMesh) {
          if (o.material) o.material.side = THREE.DoubleSide;
          o.frustumCulled = false;
        }
      });
      scene.add(root);

      setInfo(`${g.name}  size=${size.x.toFixed(2)},${size.y.toFixed(2)},${size.z.toFixed(2)}  angle=${angle}`);

      let rafId = 0;
      const animate = () => {
        rafId = requestAnimationFrame(animate);
        renderer.render(scene, cam);
      };
      animate();

      cleanup = () => {
        cancelAnimationFrame(rafId);
        renderer.dispose();
        renderer.domElement.remove();
      };
    })().catch(e => setInfo(`error: ${e.message}`));

    return () => cleanup();
  }, [id]);

  return (
    <>
      <div ref={containerRef} style={{ position: 'fixed', inset: 0, background: '#fff' }} />
      <div style={{
        position: 'fixed', top: 12, left: 12, padding: '6px 12px',
        background: 'rgba(0,0,0,0.7)', color: '#fff', borderRadius: 8,
        fontFamily: 'monospace', fontSize: 12,
      }}>{info}</div>
      <div style={{
        position: 'fixed', bottom: 12, left: 12, padding: '6px 12px',
        background: 'rgba(0,0,0,0.6)', color: '#fff', borderRadius: 8,
        fontFamily: 'monospace', fontSize: 11,
      }}>?angle=front|side|back</div>
    </>
  );
}
