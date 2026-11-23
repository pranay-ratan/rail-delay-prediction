import React, { useEffect, useRef } from 'react';

export default function Plot({ data, layout, config, style }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !window.Plotly) return;
    
    // Plotly writes computed state (axis type, range, ...) back into the
    // objects it is given. Pass deep copies so shared theme objects such as
    // DARK_LAYOUT.xaxis are never mutated and can't leak between charts.
    const mergedLayout = structuredClone({ ...layout, autosize: true });
    const mergedData = structuredClone(data);

    const mergedConfig = {
      responsive: true,
      displayModeBar: false,
      ...config,
    };

    window.Plotly.newPlot(containerRef.current, mergedData, mergedLayout, mergedConfig);

    return () => {
      if (containerRef.current && window.Plotly) {
        window.Plotly.purge(containerRef.current);
      }
    };
  }, [data, layout, config]);

  return <div ref={containerRef} style={style || { width: '100%' }} />;
}
