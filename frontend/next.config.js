module.exports = {
  // Existing config...

  // Disable static optimization
  experimental: {
    // Force dynamic rendering
    workerThreads: false,
    cpus: 1,
  },

  // Set short-lived cache
  headers: async () => [
    {
      source: "/(.*)",
      headers: [
        {
          key: "Cache-Control",
          value: "no-store, max-age=0",
        },
      ],
    },
  ],
};
