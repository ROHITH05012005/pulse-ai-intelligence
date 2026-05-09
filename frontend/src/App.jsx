import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Sparkles, Send, RefreshCw } from 'lucide-react';

const EMOTION_MAP = {
  joy: { color: '#fbbf24', icon: '😊', label: 'Joy' },
  anger: { color: '#ef4444', icon: '😡', label: 'Anger' },
  fear: { color: '#8b5cf6', icon: '😨', label: 'Fear' },
  sadness: { color: '#3b82f6', icon: '😢', label: 'Sadness' },
  surprise: { color: '#ec4899', icon: '😲', label: 'Surprise' },
  disgust: { color: '#10b981', icon: '🤢', label: 'Disgust' },
  neutral: { color: '#64748b', icon: '😐', label: 'Neutral' },
};

function App() {
  const [text, setText] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeEmotion, setActiveEmotion] = useState('neutral');

  const classifyText = useCallback(async (val) => {
    if (!val.trim()) {
      setResults(null);
      setActiveEmotion('neutral');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('/classify', { text: val });
      setResults(response.data.classifications);
      setActiveEmotion(response.data.top_emotion);
    } catch (err) {
      console.error("Classification error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced effect for real-time analysis
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      classifyText(text);
    }, 500);
    return () => clearTimeout(timeoutId);
  }, [text, classifyText]);

  useEffect(() => {
    // Update the CSS variable for the active color
    const color = EMOTION_MAP[activeEmotion]?.color || '#64748b';
    document.documentElement.style.setProperty('--active-color', color);
  }, [activeEmotion]);

  return (
    <div className="app-container">
      <div className="bg-pulse" />
      
      <motion.div 
        className="glass-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <header>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
            <div style={{ padding: '8px', background: 'var(--active-color)', borderRadius: '12px', transition: 'all 0.5s' }}>
              <Brain size={24} color="#000" />
            </div>
            <span style={{ color: 'var(--active-color)', fontWeight: 600, letterSpacing: '2px', textTransform: 'uppercase', fontSize: '0.8rem' }}>
              Pulse AI Intelligence
            </span>
          </div>
          <h1>Emotion Intelligence</h1>
          <p className="subtitle">
            Advanced text classification engine. Type below to see the AI pulse your emotions in real-time.
          </p>
        </header>

        <div className="input-wrapper">
          <textarea
            placeholder="How are you feeling? Type your thoughts here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div style={{ position: 'absolute', bottom: '20px', right: '20px', display: 'flex', gap: '10px', alignItems: 'center' }}>
            {loading && (
              <RefreshCw className="animate-spin" size={20} color="var(--active-color)" />
            )}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => classifyText(text)}
              style={{
                background: 'var(--active-color)',
                border: 'none',
                borderRadius: '12px',
                padding: '8px 16px',
                color: 'black',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                cursor: 'pointer',
                boxShadow: '0 4px 15px rgba(0,0,0,0.2)'
              }}
            >
              <Send size={16} />
              Analyze Now
            </motion.button>
          </div>
        </div>

        <AnimatePresence>
          {results && results.length > 0 && (
            <motion.div 
              className="results-container"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              {results.map((res) => (
                <motion.div 
                  key={res.label}
                  className={`emotion-pill ${activeEmotion === res.label ? 'active' : ''}`}
                  whileHover={{ scale: 1.05 }}
                >
                  <span style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
                    {EMOTION_MAP[res.label]?.icon}
                  </span>
                  <span className="label">{res.label}</span>
                  <span className="score">{(res.score * 100).toFixed(0)}%</span>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {results && results.length === 0 && !loading && (
          <div style={{ textAlign: 'center', marginTop: '2rem', padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '12px' }}>
             <p style={{ color: 'var(--active-color)' }}>The AI Brain is still waking up... please wait 10 seconds and try again!</p>
          </div>
        )}

        {!results && !loading && (
          <div style={{ textAlign: 'center', color: 'var(--text-secondary)', marginTop: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
            <Sparkles size={16} />
            <span>AI is waiting for your input...</span>
          </div>
        )}
      </motion.div>

      <footer style={{ marginTop: '2rem', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
        Built with Google Deepmind Stack • NLP Classifier v1.0
      </footer>
    </div>
  );
}

export default App;
