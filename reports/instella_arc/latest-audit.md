# Instella-MoE ARC architecture audit

Recorded: `2026-08-03T04:39:28.191364+00:00`

This audit downloads configuration, tokenizer, and remote model code only. It does not download the 16B weights.

## base

- Repository: `amd/Instella-MoE-16B-A3B-Base`
- Revision: `5ca845e88b237ca66c9c8e1f2551933a47b0daf9`
- Safetensor bytes: `31,726,245,248`
- Config class: `InstellaMoEConfig`
- Full meta model: `success`
- Tiny forward/generation: `success`

## dpo

- Repository: `amd/Instella-MoE-16B-A3B-DPO`
- Revision: `ef5a850b1e5638a98b2e28cf321a6c1b63ccde39`
- Safetensor bytes: `31,726,245,216`
- Config class: `InstellaMoEConfig`
- Full meta model: `success`
- Tiny forward/generation: `success`

## think

- Repository: `amd/Instella-MoE-16B-A3B-Think`
- Revision: `e67a4a54d81b19692ec85ea1d1c777aa5c0bfd83`
- Safetensor bytes: `31,726,245,240`
- Config class: `InstellaMoEConfig`
- Full meta model: `success`
- Tiny forward/generation: `success`

