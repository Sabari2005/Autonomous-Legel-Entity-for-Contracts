self.onmessage = function (event) {
    const processedHTML = event.data.replace(/<script[^>]*>.*?<\/script>/gi, ""); // Remove script tags for safety
    self.postMessage(processedHTML);
  };
  