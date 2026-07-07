# Welcome to the Spatial Omics Live Review

This site provides a dynamic exploration of spatial omics literature and data. Dive into the main sections below:

<style>
  /* --- Custom CSS for clickable cards --- */
  .card-container {
    display: flex; /* Use flexbox for responsive layout */
    flex-wrap: wrap; /* Allow cards to wrap to the next line on smaller screens */
    justify-content: center; /* Center the cards */
    gap: 25px; /* Space between cards */
    margin-top: 40px; /* Space above the cards */
    margin-bottom: 40px;
  }

  .card {
    background-color: var(--md-code-bg-color, #f8f8f8); /* Use a subtle background, or change to white/lightgray */
    border: 1px solid var(--md-code-hl-color, #eee); /* Light border */
    border-radius: 8px; /* Slightly rounded corners */
    padding: 25px;
    text-align: center;
    flex: 1 1 280px; /* Flex-grow, flex-shrink, flex-basis. Adjust 280px for desired minimum card width */
    max-width: 320px; /* Max width for larger screens */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    transition: transform 0.2s ease, box-shadow 0.2s ease; /* Smooth hover effect */
    text-decoration: none; /* Remove underline from the card if it's a link */
    color: inherit; /* Inherit text color */
    display: flex; /* Use flexbox for content inside card */
    flex-direction: column; /* Stack content vertically */
    justify-content: center; /* Center content vertically */
  }

  .card:hover {
    transform: translateY(-5px); /* Lift card slightly on hover */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* More prominent shadow on hover */
  }

  .card h2 {
    margin-top: 0;
    margin-bottom: 15px;
    color: var(--md-accent-fg-color); /* Use the accent color for headings */
    font-size: 1.6em; /* Slightly larger heading */
  }

  .card p {
    font-size: 1.05em;
    line-height: 1.5;
    color: var(--md-default-fg-color--light); /* Lighter text for descriptions */
  }

  /* Responsive adjustments */
  @media (max-width: 768px) {
    .card {
      flex: 1 1 100%; /* Full width on smaller screens */
      max-width: none;
    }
  }
</style>

<div class="card-container">
  <a href="book/" class="card">
    <h2>📚 The Book</h2>
    <p>Dive deep into the theoretical foundations and methodologies of spatial omics.</p>
  </a>

  <a href="datasets/" class="card">
    <h2>📊 Spatial Omics Datasets</h2>
    <p>Explore a curated collection of publicly available spatial omics datasets.</p>
  </a>

  <a href="methods/" class="card">
    <h2>🔬 Spatial Omics Methods</h2>
    <p>Review and compare various techniques and computational methods used in spatial omics.</p>
  </a>

  <a href="applications/" class="card">
    <h2>🌟 Spatial Omics Applications</h2>
    <p>Discover real-world applications of spatial omics across different biological contexts. (Interactive!)</p>
  </a>
</div>

<p style="text-align: center; margin-top: 50px;">
  This live review is continuously updated.
</p>