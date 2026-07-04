import React, { useState, useEffect, useRef } from "react";

// API base URL — configurable via VITE_API_BASE_URL so the frontend can
// point at any backend (staging, production, a teammate's machine) without
// code changes. Falls back to localhost for local development.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// ---------------- TYPES ----------------
type Result = {
  name: string;
  score: number;
  status: "Exactly Same" | "Near Duplicate" | "Different";
};

type ImagePreview = {
  file?: File;
  preview: string;
  name: string; // Added name explicitly for reliable deletion
};
// ---------------- ICONS (Same as before) ----------------
const UploadIcon = () => (
  <svg className="w-16 h-16 text-blue-500 mb-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" /></svg>
);
const FolderIcon = () => (
  <svg className="w-4 h-4 mr-1.5 inline-block align-text-bottom" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" /></svg>
);
const ImageIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" /></svg>
);
const CloseIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
);
const TrashIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
);
const ArrowRightIcon = () => (
  <svg className="w-4 h-4 ml-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
);
const BackIcon = () => (
  <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
);

// ---------------- COMPONENT ----------------
const DuplicateImg = () => {
  const [poolImages, setPoolImages] = useState<ImagePreview[]>([]);
  const [queryImage, setQueryImage] = useState<ImagePreview | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isanalyzed, setIsanalyzed] = useState(false);
  const [selectedLightboxImage, setSelectedLightboxImage] = useState<ImagePreview | null>(null);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const [uploaded,setUploaded]=useState(false);
  
  const folderInputRef = useRef<HTMLInputElement>(null);

  // 1. ON MOUNT: Clean the backend so it matches our empty state
  useEffect(() => {
    fetch(`${API_BASE}/reset`, { method: "POST" })
      .catch(err => console.error("Failed to reset backend:", err));

    return () => {
      poolImages.forEach((img) => URL.revokeObjectURL(img.preview));
      if (queryImage) URL.revokeObjectURL(queryImage.preview);
    };
  }, []);

  // 2. HELPER: Upload Pool Images Immediately
  const uploadPoolImagesToBackend = async (files: File[]) => {
    const formData = new FormData();
    setUploaded(false);
    files.forEach(f => formData.append("files", f));

    try {
      await fetch(`${API_BASE}/upload/pool`, {
        method: "POST",
        body: formData,
      });
    } catch (err) {
      console.error("Upload error:", err);
    }
    finally{
      console.log("Uploaded pool images to backend");
      setUploaded(true);
    }
  };

  const processFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const files = Array.from(fileList).filter((file) => file.type.startsWith("image/"));
    
    // Send to backend immediately
    uploadPoolImagesToBackend(files);

    const newImages = files.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
      name: file.name
    }));

    setPoolImages((prev) => [...prev, ...newImages]);
  };

  const handlePoolUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    processFiles(e.target.files);
  };

  // 3. UPDATED: Remove image from Backend immediately when clicked
  const removeImage = async (indexToRemove: number) => {
    const imgToRemove = poolImages[indexToRemove];
    
    // Call Delete API
    try {
      await fetch(`${API_BASE}/delete/pool/${imgToRemove.name}`, {
        method: "DELETE",
      });
    } catch (err) {
      console.error("Delete error:", err);
    }

    setPoolImages((prev) => {
      const newImages = prev.filter((_, index) => index !== indexToRemove);
      URL.revokeObjectURL(prev[indexToRemove].preview);
      return newImages;
    });
  };

  // 4. UPDATED: Upload Query Image Immediately (Backend replaces old one automatically)
  const handleQueryUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];

    // Upload to Backend
    const formData = new FormData();
    formData.append("file", file);
    try {
      await fetch(`${API_BASE}/upload/query`, {
        method: "POST",
        body: formData,
      });
    } catch (err) {
      console.error("Query upload error:", err);
    }

    setQueryImage({
      file,
      preview: URL.createObjectURL(file),
      name: file.name
    });
  };

  // 5. HELPER: Clear Query Image from Backend
  const removeQueryImage = async () => {
    try {
      await fetch(`${API_BASE}/delete/query`, { method: "DELETE" });
    } catch (err) { console.error(err) }
    setQueryImage(null);
  }

  const triggerFolderUpload = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (folderInputRef.current) {
      folderInputRef.current.click();
    }
  };

  // 6. UPDATED: Analyze is now a simple Trigger (files are already there)
  const checkDuplicate = async () => {
    if (!queryImage || poolImages.length === 0) {
      alert("Please upload reference images and a query image first.");
      return;
    }

    setIsAnalyzing(true);
    setIsanalyzed(true);
    setResults([]); 

    try {
      // Just GET /analyze, no body needed
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "GET", 
      });

      if (!response.ok) throw new Error(`Server error: ${response.statusText}`);

      const data = await response.json();
      setResults(data.results);

    } catch (error) {
      console.error("Error analyzing images:", error);
      alert("Analysis failed. See console for details.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleViewResultImage = (fileName: string) => {
    const imageUrl = `${API_BASE}/images/pool/${fileName}`;
    setSelectedLightboxImage({ preview: imageUrl, name: fileName });
  };

  const handleCloseLightbox = () => {
    setSelectedLightboxImage(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-gray-800 font-sans relative">
      <style>{`::-webkit-scrollbar { display: none; } html, body { -ms-overflow-style: none; scrollbar-width: none; }`}</style>

      {/* HEADER */}
      <header className="bg-gradient-to-br from-blue-600 to-blue-800 text-white pt-16 pb-20 px-4 text-center shadow-lg">
        <h1 className="text-4xl font-extrabold tracking-tight">Near Duplicate Image Detector</h1>
        <p className="mt-3 text-lg opacity-90 font-normal">Smart Ai search</p>
      </header>

      {/* MAIN CONTENT */}
      <main className="flex-1 w-full max-w-6xl mx-auto px-5 grid grid-cols-1 md:grid-cols-2 gap-8 -mt-16 mb-10">
        
        {/* PANEL 1: REFERENCE POOL */}
        <section className="bg-white rounded-2xl p-8 shadow-xl flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900">1. Reference Pool</h2>
            <span className="bg-blue-50 text-blue-600 px-3 py-1 rounded-full text-sm font-semibold">
              {poolImages.length} Ready
            </span>
          </div>

          <div className="flex-1 min-h-[320px] border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center text-center cursor-pointer relative hover:border-blue-500 hover:bg-blue-50 transition-colors bg-slate-50">
            <input
              type="file" multiple accept="image/*"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              onChange={handlePoolUpload}
            />
            <input
              type="file" ref={folderInputRef} className="hidden"
              // @ts-ignore
              webkitdirectory="" directory="" multiple
              onChange={handlePoolUpload}
            />
            <UploadIcon />
            <div className="text-slate-600 text-base mt-3"><strong>Click to upload</strong> or drag and drop</div>
            <div className="text-slate-400 text-sm mt-1">Single image or multiple files</div>
            <button
              className="mt-4 relative z-20 bg-white border border-slate-300 px-4 py-2 rounded-md text-sm text-slate-600 font-semibold hover:border-blue-500 hover:text-blue-600 shadow-sm transition-all"
              onClick={triggerFolderUpload}
            >
              <FolderIcon /> Select Folder
            </button>
          </div>

          {poolImages.length > 0 && (
            <div className="mt-5">
              <div className="grid grid-cols-5 gap-2">
                {poolImages.slice(0, 5).map((img, i) => (
                  <div key={i} className="relative group w-full aspect-square">
                    <img
                      src={img.preview} alt="preview"
                      className="w-full h-full object-cover rounded-md border border-slate-200 cursor-pointer transition-transform hover:scale-[1.02]"
                      onClick={() => setSelectedLightboxImage(img)}
                    />
                    <button
                      onClick={(e) => { e.stopPropagation(); removeImage(i); }}
                      className="absolute top-1 right-1 bg-red-500 hover:bg-red-600 text-white p-1.5 rounded-full shadow-md opacity-0 group-hover:opacity-100 transition-all duration-200 transform hover:scale-110 z-20 cursor-pointer"
                      title="Remove image"
                    >
                      <TrashIcon />
                    </button>
                  </div>
                ))}
              </div>
              {poolImages.length > 5 && (
                <button 
                  onClick={() => setIsGalleryOpen(true)}
                  className="w-full mt-3 flex items-center justify-center py-2 px-4 bg-slate-100 hover:bg-blue-50 text-blue-600 font-semibold rounded-lg transition-colors text-sm"
                >
                  Show {poolImages.length - 5} more images <ArrowRightIcon />
                </button>
              )}
            </div>
          )}
        </section>

        {/* PANEL 2: QUERY IMAGE */}
        <section className="bg-white rounded-2xl p-8 shadow-xl flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900">2. Query Image</h2>
          </div>

          {/* {!queryImage && uploaded ? (
            <div className="flex-1 min-h-[320px] border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center text-center cursor-pointer relative hover:border-blue-500 hover:bg-blue-50 transition-colors bg-slate-50">
              <input
                type="file" accept="image/*"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                onChange={handleQueryUpload}
              />
              <UploadIcon />
              <div className="text-slate-600 text-base mt-3"><strong>Select image</strong> to check</div>
            </div>
          ) : queryImage&&(
            <div className="flex-1 flex flex-col">
              <div className="relative group w-full h-[250px] mt-4 bg-slate-100 rounded-lg border border-slate-200">
                 <img
                  src={queryImage.preview} alt="Query"
                  className="w-full h-full object-contain rounded-lg cursor-pointer"
                  onClick={() => setSelectedLightboxImage(queryImage)}
                />
                 <button
                    onClick={(e) => { e.stopPropagation(); removeQueryImage(); }}
                    className="absolute top-1 right-1 bg-red-500 hover:bg-red-600 text-white p-1.5 rounded-full shadow-md opacity-0 group-hover:opacity-100 transition-all duration-200 transform hover:scale-110 z-20 cursor-pointer"
                    title="Remove image"
                  >
                    <TrashIcon />
                  </button>
              </div>
              <button
                className="text-blue-600 text-sm mt-2 underline hover:text-blue-800"
                onClick={() => removeQueryImage()}
              >
                Remove & Change
              </button>
            </div>
          ): !queryImage && !uploaded &&(
            <div className="flex-1 min-h-[320px] border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center text-center cursor-pointer relative hover:border-blue-500 hover:bg-blue-50 transition-colors bg-slate-50">
              <loading className="animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mb-4"></loading>
              <div className="text-slate-600 text-base mt-3"><strong>Upload reference images first</strong> to enable query upload</div>
            </div>
          )} */}
          {
  !uploaded && poolImages.length==0 ? (
    // 🔒 Reference images NOT uploaded
    <div className="flex-1 min-h-[320px] border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center text-center bg-slate-50">
      <div className="animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mb-4" />
      <div className="text-slate-600 text-base mt-3">
        <strong>Upload reference images first</strong> to enable query upload
      </div>
    </div>
  ) :!uploaded?(
    <div className="flex-1 min-h-[320px] border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center text-center bg-slate-50">
      <div className="animate-spin w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full mb-4" />
      <div className="text-slate-600 text-base mt-3">
        <strong>Training Pool images</strong> to enable query upload
      </div>
    </div>

  ):
  !queryImage ? (
    // 📤 Upload query image
    <div className="flex-1 min-h-[320px] border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center text-center cursor-pointer relative hover:border-blue-500 hover:bg-blue-50 transition-colors bg-slate-50">
      <input
        type="file"
        accept="image/*"
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
        onChange={handleQueryUpload}
      />
      <UploadIcon />
      <div className="text-slate-600 text-base mt-3">
        <strong>Select image</strong> to check
      </div>
    </div>
  ) : (
    // 🖼️ Query image preview
    <div className="flex-1 flex flex-col">
      <div className="relative group w-full h-[250px] mt-4 bg-slate-100 rounded-lg border border-slate-200">
        <img
          src={queryImage.preview}
          alt="Query"
          className="w-full h-full object-contain rounded-lg cursor-pointer"
          onClick={() => setSelectedLightboxImage(queryImage)}
        />

        <button
          onClick={(e: React.MouseEvent<HTMLButtonElement>) => {
            e.stopPropagation();
            removeQueryImage();
          }}
          className="absolute top-1 right-1 bg-red-500 hover:bg-red-600 text-white p-1.5 rounded-full shadow-md opacity-0 group-hover:opacity-100 transition-all duration-200 transform hover:scale-110 z-20"
          title="Remove image"
        >
          <TrashIcon />
        </button>
      </div>

      <button
        className="text-blue-600 text-sm mt-2 underline hover:text-blue-800"
        onClick={removeQueryImage}
      >
        Remove & Change
      </button>
    </div>
  )
}


          <button
            className={`w-full mt-6 py-4 rounded-xl text-white text-base font-semibold shadow-md transition-colors ${
              isAnalyzing ? "bg-slate-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
            }`}
            onClick={checkDuplicate}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? "Analyzing...": "Analyze Similarity"}
          </button>
        </section>
      </main>

      {/* RESULTS SECTION */}
      {results.length > 0 && (
        <section className="w-full max-w-6xl mx-auto px-5 mb-12">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Analysis Results</h2>
          {results.map((r, i) => {
            const colorClass = r.score > 0.9 ? "text-green-600" : r.score > 0.8 ? "text-amber-500" : "text-slate-400";
            const borderClass = r.score > 0.9 ? "border-l-green-600" : r.score > 0.8 ? "border-l-amber-500" : "border-l-slate-400";
            const bgClass = r.score > 0.9 ? "bg-green-600" : r.score > 0.8 ? "bg-amber-500" : "bg-slate-400";
            const badgeBg = r.score > 0.9 ? "bg-green-600/10" : r.score > 0.8 ? "bg-amber-500/10" : "bg-slate-400/10";
            return (
              <div key={i} className={`bg-white rounded-xl p-5 mb-4 flex items-center justify-between border-l-[5px] shadow-sm ${borderClass}`}>
                <div className="flex items-center gap-4">
                  <div 
                    className="bg-slate-100 p-2 rounded-lg text-gray-600 cursor-pointer hover:bg-slate-200 hover:text-blue-600 transition-colors"
                    onClick={() => handleViewResultImage(r.name)}
                    title="Click to view image"
                  >
                    <ImageIcon />
                  </div>
                  <div>
                    <h3 className="m-0 text-base font-medium">{r.name}</h3>
                    <span className={`text-sm font-semibold px-2 py-0.5 rounded ${colorClass} ${badgeBg}`}>{r.status}</span>
                  </div>
                </div>
                <div className="text-right w-28">
                  {/* <span className={`font-bold text-lg ${colorClass}`}>{(r.score * 100).toFixed(0)}%</span> */}
                  {/* <div className="w-full h-1.5 bg-slate-200 rounded-full mt-1.5 overflow-hidden">
                    <div className={`h-full rounded-full ${bgClass}`} style={{ width: `${r.score * 100}%` }}></div>
                  </div> */}
                </div>
              </div>
            );
          })}
         
        </section>
      )}
      {
        results.length === 0 && !isAnalyzing && isanalyzed &&(
          <section className="w-full max-w-6xl mx-auto px-5 mb-12">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Analysis Results</h2>
            <p className="text-slate-500">No matching image Found</p>
          </section>
        )
      }

      {/* FOOTER */}
      <footer className="mt-auto text-center py-8 text-slate-400 border-t border-slate-200 bg-slate-50">
        <p>© 2024 Duplicate Image Detection System</p>
      </footer>

      {/* GALLERY & LIGHTBOX (Same logic) */}
      {isGalleryOpen && (
        <div className="fixed inset-0 z-[60] bg-slate-50 overflow-y-auto animate-in fade-in duration-200">
          <div className="max-w-7xl mx-auto px-5 py-8">
            <div className="flex items-center justify-between mb-8 sticky top-0 bg-slate-50/95 backdrop-blur-sm py-4 z-10 border-b border-slate-200">
              <button onClick={() => setIsGalleryOpen(false)} className="p-2 hover:bg-white hover:shadow-sm rounded-full transition-all text-slate-600"><BackIcon /></button>
              <button onClick={() => setIsGalleryOpen(false)} className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-semibold transition-colors">Done</button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-4">
              {poolImages.map((img, i) => (
                <div key={i} className="relative group w-full aspect-square bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                   <img src={img.preview} className="w-full h-full object-cover cursor-pointer transition-transform hover:scale-105" onClick={() => setSelectedLightboxImage(img)}/>
                   <button onClick={(e) => { e.stopPropagation(); removeImage(i); }} className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white p-1.5 rounded-full shadow-md opacity-0 group-hover:opacity-100 transition-all duration-200"><TrashIcon /></button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {selectedLightboxImage && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in zoom-in duration-200" onClick={handleCloseLightbox}>
          <div className="relative w-fit h-fit flex items-center justify-center max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <img src={selectedLightboxImage.preview} className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"/>
            <button className="absolute top-4 right-4 p-2 rounded-full bg-black/50 text-white hover:bg-black/70 transition-colors cursor-pointer" onClick={handleCloseLightbox}><CloseIcon /></button>
          </div>
        </div>
      )}
    </div>
  );
}
export default DuplicateImg;
