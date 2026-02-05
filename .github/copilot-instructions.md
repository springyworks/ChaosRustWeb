<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->
- [x] Verify that the copilot-instructions.md file in the .github directory is created.

- [x] Clarify Project Requirements
    <!-- Done via web_crate_prompt.md -->

- [x] Scaffold the Project
    <!-- Created Cargo.toml, src/main.rs, index.html -->

- [x] Customize the Project
    <!-- Ported logic from wed04-1730.rs -->

- [ ] Install Required Extensions
    <!-- None explicit, Bevy handles most -->

- [ ] Compile the Project
    <!-- Will verify with cargo check -->

- [ ] Launch the Project
    <!-- Provide instructions -->

-[ ] Ask user for feedback, after they have run the project, to ensure it meets their expectations.
    <!-- Prompt for feedback -->
 [ ] Ensure Documentation is Complete

    <!-- Verify README.md -->

<!--
## Execution Guidelines
... (Standard guidelines)
-->


Evaluate the possibility to run the burn crate as alternative to simple approach , which is fine to  , it can do webgpu in browser with WASM⚠️ Warning When using one of the wgpu backends, you may encounter compilation errors related to recursive type evaluation. This is due to complex type nesting within the wgpu dependency chain. To resolve this issue, add the following line at the top of your main.rs or lib.rs file:

#![recursion_limit = "256"]
The default recursion limit (128) is often just below the required depth (typically 130-150) due to deeply nested associated types and trait bounds.