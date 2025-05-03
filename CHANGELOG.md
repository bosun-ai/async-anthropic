# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0](https://github.com/bosun-ai/async-anthropic/compare/v0.5.0...v0.6.0) - 2025-05-03

### Added

- Track input tokens when streaming

## [0.5.0](https://github.com/bosun-ai/async-anthropic/compare/v0.4.0...v0.5.0) - 2025-05-02

### Added

- Support streaming messages with tool use ([#10](https://github.com/bosun-ai/async-anthropic/pull/10))

### Other

- *(deps)* Bump reqwest from 0.12.9 to 0.12.15 in the minor group ([#9](https://github.com/bosun-ai/async-anthropic/pull/9))

## [0.4.0](https://github.com/bosun-ai/async-anthropic/compare/v0.3.0...v0.4.0) - 2025-04-27

### Added

- Add support for models api
- Convenience helpers for accessing message content
- Implement streaming for messages api

### Other

- *(deps)* Bump the minor group with 7 updates ([#8](https://github.com/bosun-ai/async-anthropic/pull/8))
- *(ci)* Add dependabot.yml

## [0.3.0](https://github.com/bosun-ai/async-anthropic/compare/v0.2.1...v0.3.0) - 2025-02-18

### Added

- Add backoff implementation (#5)

### Other

- Add backoff as a major feature

## [0.2.1](https://github.com/bosun-ai/async-anthropic/compare/v0.2.0...v0.2.1) - 2025-02-10

### Added

- Inner content of message list must be public
- Also implement deref mut for message content list

## [0.2.0](https://github.com/bosun-ai/async-anthropic/compare/v0.1.0...v0.2.0) - 2025-02-10

### Added

- Ensure all message content is accessible
- Add convenience method to access first matching text content in message
- Add convenience str to message conversion and partialeq

### Other

- release v0.1.0 (#2)

## [0.1.0](https://github.com/bosun-ai/async-anthropic/releases/tag/v0.1.0) - 2025-02-09

### Added

- Client rewrite

### Fixed

- Add tests

