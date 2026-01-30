//! HTML Dashboard Generator
//!
//! Generates interactive HTML dashboards for MQS results.

use apr_qa_runner::EvidenceCollector;

use crate::error::Result;
use crate::mqs::MqsScore;
use crate::popperian::PopperianScore;

/// HTML dashboard generator
#[derive(Debug, Default)]
pub struct HtmlDashboard {
    /// Dashboard title
    title: String,
    /// Include interactive charts
    include_charts: bool,
}

impl HtmlDashboard {
    /// Create a new dashboard generator
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            include_charts: true,
        }
    }

    /// Disable interactive charts
    #[must_use]
    pub fn without_charts(mut self) -> Self {
        self.include_charts = false;
        self
    }

    /// Generate HTML dashboard
    ///
    /// # Errors
    ///
    /// Returns an error if HTML generation fails.
    pub fn generate(
        &self,
        mqs: &MqsScore,
        popperian: &PopperianScore,
        _evidence: &EvidenceCollector,
    ) -> Result<String> {
        let grade_color = Self::grade_color(&mqs.grade);
        let pass_rate = if mqs.total_tests > 0 {
            (mqs.tests_passed as f64 / mqs.total_tests as f64) * 100.0
        } else {
            0.0
        };

        let html = format!(
            r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #0f3460;
            --success: #00d26a;
            --warning: #ffc107;
            --danger: #ff4757;
            --grade-color: {grade_color};
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ margin-bottom: 1rem; color: #fff; }}
        .model-id {{ color: #888; font-size: 0.9em; }}
        .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .card h3 {{ color: #aaa; font-size: 0.85em; text-transform: uppercase; margin-bottom: 0.5rem; }}
        .score-large {{
            font-size: 3rem;
            font-weight: bold;
            color: var(--grade-color);
        }}
        .grade {{ font-size: 4rem; font-weight: bold; color: var(--grade-color); }}
        .stat {{ display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        .stat:last-child {{ border-bottom: none; }}
        .stat-label {{ color: #888; }}
        .stat-value {{ font-weight: 600; }}
        .progress-bar {{ background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden; margin-top: 0.5rem; }}
        .progress-fill {{ height: 100%; transition: width 0.3s; }}
        .success {{ background: var(--success); }}
        .warning {{ background: var(--warning); }}
        .danger {{ background: var(--danger); }}
        .gateway {{ display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0; }}
        .gateway-icon {{ width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; }}
        .gateway-pass {{ background: var(--success); color: #000; }}
        .gateway-fail {{ background: var(--danger); color: #fff; }}
        .category-bar {{ display: flex; align-items: center; gap: 1rem; margin: 0.75rem 0; }}
        .category-name {{ width: 60px; font-size: 0.85em; color: #888; }}
        .category-track {{ flex: 1; background: rgba(255,255,255,0.1); border-radius: 4px; height: 12px; overflow: hidden; }}
        .category-fill {{ height: 100%; background: linear-gradient(90deg, var(--success), #00ff88); border-radius: 4px; }}
        .category-value {{ width: 50px; text-align: right; font-size: 0.85em; }}
        .falsification {{ background: rgba(255,71,87,0.1); border-left: 3px solid var(--danger); padding: 0.75rem; margin: 0.5rem 0; border-radius: 0 4px 4px 0; }}
        .falsification-gate {{ font-weight: 600; color: var(--danger); }}
        .black-swan {{ background: rgba(255,71,87,0.2); }}
        .timestamp {{ color: #666; font-size: 0.8em; margin-top: 2rem; text-align: center; }}
        @media (max-width: 600px) {{
            body {{ padding: 1rem; }}
            .score-large {{ font-size: 2rem; }}
            .grade {{ font-size: 3rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="model-id">Model: {model_id}</p>

        <div class="dashboard">
            <!-- MQS Score Card -->
            <div class="card">
                <h3>MQS Score</h3>
                <div class="score-large">{normalized_score:.1}</div>
                <div class="stat">
                    <span class="stat-label">Raw Score</span>
                    <span class="stat-value">{raw_score}/1000</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Status</span>
                    <span class="stat-value">{qualification_status}</span>
                </div>
            </div>

            <!-- Grade Card -->
            <div class="card">
                <h3>Grade</h3>
                <div class="grade">{grade}</div>
                <div class="stat">
                    <span class="stat-label">Production Ready</span>
                    <span class="stat-value">{production_ready}</span>
                </div>
            </div>

            <!-- Pass Rate Card -->
            <div class="card">
                <h3>Test Results</h3>
                <div class="stat">
                    <span class="stat-label">Total Tests</span>
                    <span class="stat-value">{total_tests}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Passed</span>
                    <span class="stat-value" style="color: var(--success)">{tests_passed}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Failed</span>
                    <span class="stat-value" style="color: var(--danger)">{tests_failed}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {pass_rate_class}" style="width: {pass_rate:.1}%"></div>
                </div>
            </div>

            <!-- Gateways Card -->
            <div class="card">
                <h3>Gateway Checks</h3>
                {gateway_html}
            </div>

            <!-- Categories Card -->
            <div class="card" style="grid-column: span 2;">
                <h3>Category Breakdown</h3>
                {categories_html}
            </div>

            <!-- Popperian Card -->
            <div class="card">
                <h3>Popperian Analysis</h3>
                <div class="stat">
                    <span class="stat-label">Hypotheses Tested</span>
                    <span class="stat-value">{hypotheses_tested}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Corroborated</span>
                    <span class="stat-value">{corroborated}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Falsified</span>
                    <span class="stat-value">{falsified}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Black Swans</span>
                    <span class="stat-value">{black_swans}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Confidence</span>
                    <span class="stat-value">{confidence:.1}%</span>
                </div>
            </div>

            <!-- Falsifications Card -->
            <div class="card" style="grid-column: span 2;">
                <h3>Falsifications</h3>
                {falsifications_html}
            </div>
        </div>

        <p class="timestamp">Generated: {timestamp}</p>
    </div>
</body>
</html>"##,
            title = self.title,
            model_id = mqs.model_id,
            grade_color = grade_color,
            normalized_score = mqs.normalized_score,
            raw_score = mqs.raw_score,
            grade = mqs.grade,
            qualification_status = if mqs.qualifies() {
                "Qualified"
            } else {
                "Not Qualified"
            },
            production_ready = if mqs.is_production_ready() {
                "Yes"
            } else {
                "No"
            },
            total_tests = mqs.total_tests,
            tests_passed = mqs.tests_passed,
            tests_failed = mqs.tests_failed,
            pass_rate = pass_rate,
            pass_rate_class = if pass_rate >= 90.0 {
                "success"
            } else if pass_rate >= 70.0 {
                "warning"
            } else {
                "danger"
            },
            gateway_html = self.render_gateways(mqs),
            categories_html = self.render_categories(mqs),
            hypotheses_tested = popperian.hypotheses_tested,
            corroborated = popperian.corroborated,
            falsified = popperian.falsified,
            black_swans = popperian.black_swan_count,
            confidence = popperian.confidence_level * 100.0,
            falsifications_html = self.render_falsifications(popperian),
            timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        );

        Ok(html)
    }

    /// Render gateway checks HTML
    fn render_gateways(&self, mqs: &MqsScore) -> String {
        let mut html = String::new();
        for gateway in &mqs.gateways {
            let (icon_class, icon) = if gateway.passed {
                ("gateway-pass", "✓")
            } else {
                ("gateway-fail", "✗")
            };
            html.push_str(&format!(
                r#"<div class="gateway">
                    <div class="gateway-icon {}">{}</div>
                    <span>{}: {}</span>
                </div>"#,
                icon_class, icon, gateway.id, gateway.description
            ));
        }
        if html.is_empty() {
            html = "<p>No gateway checks recorded</p>".to_string();
        }
        html
    }

    /// Render category breakdown HTML
    fn render_categories(&self, mqs: &MqsScore) -> String {
        let categories = [
            ("QUAL", mqs.categories.qual, 200),
            ("PERF", mqs.categories.perf, 150),
            ("STAB", mqs.categories.stab, 200),
            ("COMP", mqs.categories.comp, 150),
            ("EDGE", mqs.categories.edge, 150),
            ("REGR", mqs.categories.regr, 150),
        ];

        let mut html = String::new();
        for (name, score, max) in categories {
            let pct = if max > 0 {
                (score as f64 / max as f64) * 100.0
            } else {
                0.0
            };
            html.push_str(&format!(
                r#"<div class="category-bar">
                    <span class="category-name">{}</span>
                    <div class="category-track">
                        <div class="category-fill" style="width: {:.1}%"></div>
                    </div>
                    <span class="category-value">{}/{}</span>
                </div>"#,
                name, pct, score, max
            ));
        }
        html
    }

    /// Render falsifications HTML
    fn render_falsifications(&self, popperian: &PopperianScore) -> String {
        if popperian.falsifications.is_empty() {
            return "<p style=\"color: var(--success)\">No falsifications - all hypotheses corroborated!</p>".to_string();
        }

        let mut html = String::new();
        for f in popperian.falsifications.iter().take(10) {
            let class = if f.is_black_swan {
                "falsification black-swan"
            } else {
                "falsification"
            };
            html.push_str(&format!(
                r#"<div class="{}">
                    <span class="falsification-gate">{}</span>
                    {}: {}
                    {}
                </div>"#,
                class,
                f.gate_id,
                f.hypothesis,
                Self::escape_html(&f.evidence),
                if f.is_black_swan {
                    " <strong>(Black Swan)</strong>"
                } else {
                    ""
                }
            ));
        }

        if popperian.falsifications.len() > 10 {
            html.push_str(&format!(
                "<p>... and {} more falsifications</p>",
                popperian.falsifications.len() - 10
            ));
        }

        html
    }

    /// Get color for grade
    fn grade_color(grade: &str) -> &'static str {
        match grade {
            "A+" | "A" | "A-" => "#00d26a",
            "B+" | "B" | "B-" => "#7bed9f",
            "C+" | "C" | "C-" => "#ffc107",
            "D+" | "D" | "D-" => "#ff9f43",
            _ => "#ff4757",
        }
    }

    /// Escape HTML special characters
    fn escape_html(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mqs::{CategoryScores, GatewayResult};
    use crate::popperian::FalsificationDetail;

    fn test_mqs() -> MqsScore {
        MqsScore {
            model_id: "test/model".to_string(),
            raw_score: 850,
            normalized_score: 92.5,
            grade: "A-".to_string(),
            gateways: vec![
                GatewayResult::passed("G1", "Model loads"),
                GatewayResult::passed("G2", "Inference works"),
                GatewayResult::passed("G3", "No crashes"),
                GatewayResult::passed("G4", "Output valid"),
            ],
            gateways_passed: true,
            categories: CategoryScores {
                qual: 180,
                perf: 130,
                stab: 170,
                comp: 130,
                edge: 120,
                regr: 120,
            },
            total_tests: 100,
            tests_passed: 95,
            tests_failed: 5,
            penalties: vec![],
            total_penalty: 0,
        }
    }

    fn test_popperian() -> PopperianScore {
        PopperianScore {
            model_id: "test/model".to_string(),
            hypotheses_tested: 100,
            corroborated: 95,
            falsified: 5,
            inconclusive: 0,
            corroboration_ratio: 0.95,
            severity_weighted_score: 0.93,
            confidence_level: 0.92,
            reproducibility_index: 0.85,
            black_swan_count: 0,
            falsifications: vec![FalsificationDetail {
                gate_id: "F-EDGE-001".to_string(),
                hypothesis: "Handles empty input".to_string(),
                evidence: "Returned garbage".to_string(),
                severity: 3,
                is_black_swan: false,
                occurrence_count: 1,
            }],
        }
    }

    #[test]
    fn test_html_generation() {
        let dashboard = HtmlDashboard::new("Test Dashboard");
        let collector = EvidenceCollector::new();

        let html = dashboard
            .generate(&test_mqs(), &test_popperian(), &collector)
            .expect("Failed to generate");

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Dashboard"));
        assert!(html.contains("test/model"));
        assert!(html.contains("A-"));
        assert!(html.contains("92.5"));
    }

    #[test]
    fn test_grade_colors() {
        assert_eq!(HtmlDashboard::grade_color("A+"), "#00d26a");
        assert_eq!(HtmlDashboard::grade_color("B"), "#7bed9f");
        assert_eq!(HtmlDashboard::grade_color("C"), "#ffc107");
        assert_eq!(HtmlDashboard::grade_color("F"), "#ff4757");
    }

    #[test]
    fn test_gateway_rendering() {
        let dashboard = HtmlDashboard::new("Test");
        let mqs = test_mqs();

        let html = dashboard.render_gateways(&mqs);

        assert!(html.contains("G1"));
        assert!(html.contains("gateway-pass"));
        assert!(html.contains("Model loads"));
    }

    #[test]
    fn test_category_rendering() {
        let dashboard = HtmlDashboard::new("Test");
        let mqs = test_mqs();

        let html = dashboard.render_categories(&mqs);

        assert!(html.contains("QUAL"));
        assert!(html.contains("PERF"));
        assert!(html.contains("180/200"));
    }

    #[test]
    fn test_falsification_rendering() {
        let dashboard = HtmlDashboard::new("Test");
        let popperian = test_popperian();

        let html = dashboard.render_falsifications(&popperian);

        assert!(html.contains("F-EDGE-001"));
        assert!(html.contains("empty input"));
    }

    #[test]
    fn test_html_escaping() {
        assert_eq!(HtmlDashboard::escape_html("<script>"), "&lt;script&gt;");
    }

    #[test]
    fn test_html_escaping_ampersand() {
        assert_eq!(HtmlDashboard::escape_html("a & b"), "a &amp; b");
    }

    #[test]
    fn test_html_escaping_quotes() {
        assert_eq!(
            HtmlDashboard::escape_html("say \"hi\""),
            "say &quot;hi&quot;"
        );
    }

    #[test]
    fn test_grade_color_d() {
        assert_eq!(HtmlDashboard::grade_color("D"), "#ff9f43");
        assert_eq!(HtmlDashboard::grade_color("D+"), "#ff9f43");
    }

    #[test]
    fn test_grade_color_b_variants() {
        assert_eq!(HtmlDashboard::grade_color("B+"), "#7bed9f");
        assert_eq!(HtmlDashboard::grade_color("B-"), "#7bed9f");
    }

    #[test]
    fn test_grade_color_c_variants() {
        assert_eq!(HtmlDashboard::grade_color("C+"), "#ffc107");
        assert_eq!(HtmlDashboard::grade_color("C-"), "#ffc107");
    }

    #[test]
    fn test_gateway_failed_rendering() {
        let dashboard = HtmlDashboard::new("Test");
        let mut mqs = test_mqs();
        mqs.gateways = vec![GatewayResult::failed("G1", "Model loads", "OOM")];
        mqs.gateways_passed = false;

        let html = dashboard.render_gateways(&mqs);
        assert!(html.contains("gateway-fail"));
    }

    #[test]
    fn test_html_dashboard_default_title() {
        let dashboard = HtmlDashboard::new("MQS Report");
        let collector = EvidenceCollector::new();

        let html = dashboard
            .generate(&test_mqs(), &test_popperian(), &collector)
            .expect("Failed");

        assert!(html.contains("MQS Report"));
    }

    #[test]
    fn test_popperian_with_black_swan_rendering() {
        let dashboard = HtmlDashboard::new("Test");
        let popperian = PopperianScore {
            model_id: "test".to_string(),
            hypotheses_tested: 100,
            corroborated: 99,
            falsified: 1,
            inconclusive: 0,
            corroboration_ratio: 0.99,
            severity_weighted_score: 0.99,
            confidence_level: 0.95,
            reproducibility_index: 1.0,
            black_swan_count: 1,
            falsifications: vec![FalsificationDetail {
                gate_id: "G1-CRASH".to_string(),
                hypothesis: "No crash".to_string(),
                evidence: "SIGSEGV".to_string(),
                severity: 5,
                is_black_swan: true,
                occurrence_count: 1,
            }],
        };

        let html = dashboard.render_falsifications(&popperian);
        assert!(html.contains("G1-CRASH"));
    }
}
