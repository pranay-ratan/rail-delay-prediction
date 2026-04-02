import React, { useEffect, useRef } from 'react';

export default function Plot({ data, layout, config, style }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !window.Plotly) return;
    
    const mergedLayout = {
      ...layout,
      autosize: true,
    };

    const mergedConfig = {
      responsive: true,
      displayModeBar: false,
      ...config,
    };

    window.Plotly.newPlot(containerRef.current, data, mergedLayout, mergedConfig);

    return () => {
      if (containerRef.current && window.Plotly) {
        window.Plotly.purge(containerRef.current);
      }
    };
  }, [data, layout, config]);

  return <div ref={containerRef} style={style || { width: '100%' }} />;
}
