"""
test_report_generator.py
========================
Unit tests verifying the ReportGenerator contract:
Input:  regional_features tensor [B, embed_dim] + region_label
Output: clinical finding string + patient-friendly translation
"""

import unittest
import torch

from src.report_generator.generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ReportGenerator()

    def test_generate_finding(self):
        dummy_feat = torch.randn(1, 512)
        finding = self.generator.generate_finding(dummy_feat, region_label="heart")

        self.assertIsInstance(finding, str)
        self.assertTrue(len(finding) > 10)
        self.assertIn("cardiomediastinal", finding.lower())

    def test_patient_friendly_translation(self):
        clinical_text = "The cardiomediastinal silhouette is normal in size and contour."
        patient_text = self.generator.translate_to_patient_friendly(clinical_text)

        self.assertIsInstance(patient_text, str)
        self.assertTrue(len(patient_text) > 10)
        # Should be accessible, non-technical English
        self.assertIn("heart", patient_text.lower())

    def test_forward_dict_output(self):
        dummy_feat = torch.randn(1, 512)
        result = self.generator(dummy_feat, region_label="pleura")

        self.assertIn("clinical_finding", result)
        self.assertIn("patient_translation", result)
        self.assertIsInstance(result["clinical_finding"], str)
        self.assertIsInstance(result["patient_translation"], str)


if __name__ == "__main__":
    unittest.main()
