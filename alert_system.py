"""
Mobile Alert System for Disaster Management
==========================================

SMS and push notification system for emergency alerts and resource coordination
"""

import pandas as pd
import json
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Optional
from disaster_prediction_model import DisasterPredictionModel
from resource_planner import DisasterResourcePlanner

class DisasterAlertSystem:
    def __init__(self, model: DisasterPredictionModel, planner: DisasterResourcePlanner):
        """Initialize alert system with prediction model and resource planner"""
        self.model = model
        self.planner = planner
        self.alert_levels = {
            1: {"level": "LOW", "color": "green", "action": "Monitor"},
            2: {"level": "MODERATE", "color": "yellow", "action": "Prepare"},
            3: {"level": "HIGH", "color": "orange", "action": "Respond"},
            4: {"level": "SEVERE", "color": "red", "action": "Evacuate"},
            5: {"level": "EXTREME", "color": "darkred", "action": "Emergency"}
        }
        
        # Sample contact database
        self.emergency_contacts = self._initialize_contacts()
        self.sent_alerts = []
    
    def _initialize_contacts(self):
        """Initialize emergency contact database"""
        return {
            'disaster_management': [
                {'name': 'NDRF Control Room', 'phone': '+91-11-26701728', 'role': 'National Response'},
                {'name': 'State Emergency Operations', 'phone': '+91-XXX-XXXXXXX', 'role': 'State Coordination'},
                {'name': 'District Collector', 'phone': '+91-XXX-XXXXXXX', 'role': 'District Administration'}
            ],
            'medical_emergency': [
                {'name': 'Emergency Medical Services', 'phone': '108', 'role': 'Medical Response'},
                {'name': 'Blood Bank Emergency', 'phone': '+91-XXX-XXXXXXX', 'role': 'Blood Supply'},
                {'name': 'Hospital Command Center', 'phone': '+91-XXX-XXXXXXX', 'role': 'Medical Coordination'}
            ],
            'police_fire': [
                {'name': 'Police Control Room', 'phone': '100', 'role': 'Law & Order'},
                {'name': 'Fire Emergency', 'phone': '101', 'role': 'Fire Response'},
                {'name': 'Women Helpline', 'phone': '1091', 'role': 'Women Safety'}
            ],
            'utilities': [
                {'name': 'Electricity Emergency', 'phone': '+91-XXX-XXXXXXX', 'role': 'Power Restoration'},
                {'name': 'Water Supply Emergency', 'phone': '+91-XXX-XXXXXXX', 'role': 'Water Services'},
                {'name': 'Telecom Emergency', 'phone': '+91-XXX-XXXXXXX', 'role': 'Communication'}
            ],
            'public_numbers': [
                {'name': 'Disaster Helpline', 'phone': '1078', 'role': 'General Help'},
                {'name': 'Child Helpline', 'phone': '1098', 'role': 'Child Safety'},
                {'name': 'Senior Citizens Helpline', 'phone': '14567', 'role': 'Elder Care'}
            ]
        }
    
    def generate_alert_message(self, state: str, disaster_type: str, area: float, 
                              severity: int, predictions: Dict) -> Dict:
        """Generate comprehensive alert message"""
        
        alert_info = self.alert_levels[severity]
        current_time = datetime.now()
        
        # Base alert message
        message = f"""
🚨 DISASTER ALERT - {alert_info['level']} LEVEL 🚨

📍 LOCATION: {state}
🌪️ DISASTER: {disaster_type}
📏 AREA: {area:,} sq km
⚠️ SEVERITY: {severity}/5 ({alert_info['level']})
🕐 TIME: {current_time.strftime('%Y-%m-%d %H:%M:%S')}

👥 PREDICTED IMPACT:
• Deaths: {predictions['Deaths']:,}
• Injured: {predictions['Injured']:,}
• Affected: {predictions['Affected']:,}
• Total Victims: {predictions['Total_Victims']:,}

🎯 RECOMMENDED ACTION: {alert_info['action']}
"""
        
        # Add specific instructions based on disaster type and severity
        if disaster_type.lower() == 'earthquake' and severity >= 4:
            message += """
🏠 EARTHQUAKE SAFETY:
• Drop, Cover, Hold On
• Stay away from windows/glass
• Exit building if safe to do so
• Check for gas leaks after shaking stops
"""
        
        elif disaster_type.lower() == 'flood' and severity >= 3:
            message += """
🌊 FLOOD SAFETY:
• Move to higher ground immediately
• Avoid walking/driving through flood water
• Stay away from electrical lines
• Listen to emergency broadcasts
"""
        
        elif disaster_type.lower() == 'tropical cyclone' and severity >= 3:
            message += """
🌀 CYCLONE SAFETY:
• Stay indoors, away from windows
• Secure loose objects outside
• Keep emergency supplies ready
• Monitor weather updates
"""
        
        # Emergency contacts
        message += "\n📞 EMERGENCY CONTACTS:\n"
        message += "• Police: 100 | Fire: 101 | Medical: 108\n"
        message += "• Disaster Helpline: 1078\n"
        message += "• NDRF: +91-11-26701728\n"
        
        message += "\n⚡ Stay Safe! Share this alert with others."
        
        return {
            'message': message,
            'alert_level': alert_info['level'],
            'severity': severity,
            'timestamp': current_time,
            'disaster_type': disaster_type,
            'location': state,
            'recipients': self._get_alert_recipients(severity)
        }
    
    def _get_alert_recipients(self, severity: int) -> List[str]:
        """Get list of recipients based on alert severity"""
        recipients = []
        
        if severity >= 5:  # EXTREME
            recipients.extend(['disaster_management', 'medical_emergency', 'police_fire', 'utilities'])
        elif severity >= 4:  # SEVERE
            recipients.extend(['disaster_management', 'medical_emergency', 'police_fire'])
        elif severity >= 3:  # HIGH
            recipients.extend(['disaster_management', 'medical_emergency'])
        elif severity >= 2:  # MODERATE
            recipients.extend(['disaster_management'])
        
        return recipients
    
    def send_sms_alert(self, phone_number: str, message: str) -> Dict:
        """Simulate SMS sending (in real implementation, integrate with SMS gateway)"""
        
        # Simulate SMS API call
        sms_response = {
            'status': 'sent',
            'message_id': f"SMS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'phone': phone_number,
            'timestamp': datetime.now(),
            'characters': len(message),
            'cost': len(message) * 0.05  # Simulated cost
        }
        
        return sms_response
    
    def send_push_notification(self, device_tokens: List[str], title: str, body: str) -> Dict:
        """Simulate push notification (in real implementation, integrate with FCM/APNS)"""
        
        # Simulate push notification API call
        push_response = {
            'status': 'sent',
            'notification_id': f"PUSH_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'devices_reached': len(device_tokens),
            'timestamp': datetime.now(),
            'title': title,
            'body': body[:100] + "..." if len(body) > 100 else body
        }
        
        return push_response
    
    def broadcast_alert(self, state: str, disaster_type: str, area: float, severity: int) -> Dict:
        """Broadcast comprehensive disaster alert"""
        
        print(f"🚨 BROADCASTING DISASTER ALERT")
        print("=" * 50)
        print(f"Location: {state}")
        print(f"Disaster: {disaster_type}")
        print(f"Severity: {severity}/5")
        
        # Generate predictions
        predictions = self.model.predict_victims(state, area, disaster_type, severity)
        ensemble = predictions['ensemble']
        
        # Generate alert message
        alert = self.generate_alert_message(state, disaster_type, area, severity, ensemble)
        
        # Simulate broadcasting to different channels
        broadcast_results = {
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now(),
            'alert_level': alert['alert_level'],
            'location': state,
            'disaster_type': disaster_type,
            'channels': {}
        }
        
        # SMS Broadcasting
        sms_results = []
        total_sms_sent = 0
        
        for recipient_group in alert['recipients']:
            if recipient_group in self.emergency_contacts:
                for contact in self.emergency_contacts[recipient_group]:
                    sms_result = self.send_sms_alert(contact['phone'], alert['message'])
                    sms_results.append({
                        'contact': contact['name'],
                        'phone': contact['phone'],
                        'role': contact['role'],
                        'result': sms_result
                    })
                    total_sms_sent += 1
        
        broadcast_results['channels']['sms'] = {
            'total_sent': total_sms_sent,
            'results': sms_results
        }
        
        # Push Notification Broadcasting
        # Simulate device tokens for different user groups
        device_groups = {
            'emergency_responders': [f"device_token_{i}" for i in range(50)],
            'public_safety_officers': [f"device_token_{i}" for i in range(100, 200)],
            'government_officials': [f"device_token_{i}" for i in range(200, 250)]
        }
        
        push_results = []
        total_push_sent = 0
        
        for group, tokens in device_groups.items():
            if severity >= 3 or group == 'emergency_responders':  # Send to public for high severity
                push_result = self.send_push_notification(
                    tokens,
                    f"🚨 {alert['alert_level']} ALERT - {disaster_type}",
                    f"{state}: {disaster_type} (Severity {severity}/5). {ensemble['Total_Victims']:,} victims predicted."
                )
                push_results.append({
                    'group': group,
                    'devices': len(tokens),
                    'result': push_result
                })
                total_push_sent += len(tokens)
        
        broadcast_results['channels']['push_notifications'] = {
            'total_sent': total_push_sent,
            'results': push_results
        }
        
        # Social Media Broadcasting (simulated)
        if severity >= 4:
            social_media_results = {
                'twitter': {'status': 'posted', 'reach': 10000, 'engagement': 500},
                'facebook': {'status': 'posted', 'reach': 15000, 'engagement': 800},
                'instagram': {'status': 'posted', 'reach': 8000, 'engagement': 300}
            }
            broadcast_results['channels']['social_media'] = social_media_results
        
        # Radio/TV Broadcasting (simulated)
        if severity >= 4:
            broadcast_media_results = {
                'all_india_radio': {'status': 'broadcasted', 'reach': 100000},
                'doordarshan': {'status': 'broadcasted', 'reach': 150000},
                'private_news_channels': {'status': 'alerted', 'channels': 25}
            }
            broadcast_results['channels']['broadcast_media'] = broadcast_media_results
        
        # Store alert in history
        self.sent_alerts.append(broadcast_results)
        
        print(f"✅ Alert broadcast complete!")
        print(f"   SMS sent: {total_sms_sent}")
        print(f"   Push notifications: {total_push_sent}")
        print(f"   Alert ID: {broadcast_results['alert_id']}")
        
        return broadcast_results
    
    def generate_resource_alert(self, resource_plan: Dict) -> Dict:
        """Generate resource requirement alert for suppliers and agencies"""
        
        resources = resource_plan['resources']
        costs = resource_plan['costs']
        
        resource_message = f"""
📦 RESOURCE REQUIREMENT ALERT

🕐 TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💰 TOTAL BUDGET: ₹{costs['grand_total']:,.2f}

🚑 URGENT REQUIREMENTS:
"""
        
        # Prioritize critical resources
        critical_resources = []
        
        if 'medical' in resources:
            medical = resources['medical']
            if medical.get('doctors', 0) > 0:
                critical_resources.append(f"• Doctors: {medical['doctors']:,}")
            if medical.get('ambulances', 0) > 0:
                critical_resources.append(f"• Ambulances: {medical['ambulances']:,}")
        
        if 'personnel' in resources:
            personnel = resources['personnel']
            if personnel.get('search_rescue_specialists', 0) > 0:
                critical_resources.append(f"• Search & Rescue Teams: {personnel['search_rescue_specialists']:,}")
        
        resource_message += "\n".join(critical_resources[:5])  # Top 5 critical resources
        
        resource_message += f"""

📞 SUPPLIER CONTACTS ACTIVATED
⏱️ EXPECTED DELIVERY: As per procurement timeline
🎯 DEPLOYMENT: Multiple distribution points

For detailed requirements, contact Resource Coordination Center.
"""
        
        return {
            'message': resource_message,
            'timestamp': datetime.now(),
            'total_cost': costs['grand_total'],
            'resource_categories': list(resources.keys())
        }
    
    def send_follow_up_alerts(self, alert_id: str, status_update: str) -> Dict:
        """Send follow-up alerts with status updates"""
        
        follow_up_message = f"""
📢 DISASTER ALERT UPDATE

🆔 ALERT ID: {alert_id}
🕐 UPDATE TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 STATUS UPDATE:
{status_update}

🔄 This is a follow-up to the earlier disaster alert.
Stay tuned for further updates.

Emergency Contacts: Police-100, Fire-101, Medical-108
"""
        
        # Simulate sending follow-up
        follow_up_result = {
            'alert_id': alert_id,
            'follow_up_id': f"FOLLOWUP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'status': 'sent',
            'timestamp': datetime.now(),
            'message': follow_up_message
        }
        
        return follow_up_result
    
    def get_alert_history(self, days: int = 7) -> List[Dict]:
        """Get alert history for specified number of days"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [
            alert for alert in self.sent_alerts 
            if alert['timestamp'] >= cutoff_date
        ]
        
        return recent_alerts
    
    def generate_alert_statistics(self) -> Dict:
        """Generate statistics about sent alerts"""
        
        if not self.sent_alerts:
            return {'total_alerts': 0, 'message': 'No alerts sent yet'}
        
        total_alerts = len(self.sent_alerts)
        
        # Count by severity
        severity_counts = {}
        for alert in self.sent_alerts:
            level = alert.get('alert_level', 'UNKNOWN')
            severity_counts[level] = severity_counts.get(level, 0) + 1
        
        # Count by disaster type
        disaster_counts = {}
        for alert in self.sent_alerts:
            disaster = alert.get('disaster_type', 'UNKNOWN')
            disaster_counts[disaster] = disaster_counts.get(disaster, 0) + 1
        
        # Calculate totals
        total_sms = sum(
            alert['channels'].get('sms', {}).get('total_sent', 0) 
            for alert in self.sent_alerts
        )
        
        total_push = sum(
            alert['channels'].get('push_notifications', {}).get('total_sent', 0)
            for alert in self.sent_alerts
        )
        
        return {
            'total_alerts': total_alerts,
            'severity_breakdown': severity_counts,
            'disaster_type_breakdown': disaster_counts,
            'total_sms_sent': total_sms,
            'total_push_notifications': total_push,
            'last_alert': self.sent_alerts[-1]['timestamp'] if self.sent_alerts else None
        }

def main():
    """Demonstrate alert system functionality"""
    print("🚨 DISASTER ALERT SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize system
    data_path = r"c:\Clg projects\SIH\public_emdat_custom_request_2025-09-23_a3fd530f-e94c-4921-85e6-df85a531b149.csv"
    model = DisasterPredictionModel(data_path)
    model.load_model("disaster_prediction_model.pkl")
    
    planner = DisasterResourcePlanner(model)
    alert_system = DisasterAlertSystem(model, planner)
    
    # Demo: Broadcast severe flood alert
    flood_alert = alert_system.broadcast_alert(
        state="West Bengal",
        disaster_type="Flood",
        area=2500,
        severity=4
    )
    
    print(f"\n📱 ALERT BROADCASTED:")
    print(f"Alert ID: {flood_alert['alert_id']}")
    print(f"SMS sent: {flood_alert['channels']['sms']['total_sent']}")
    print(f"Push notifications: {flood_alert['channels']['push_notifications']['total_sent']}")
    
    # Demo: Generate resource alert
    resource_plan = planner.generate_resource_plan("West Bengal", "Flood", 2500, 4)
    resource_alert = alert_system.generate_resource_alert(resource_plan)
    
    print(f"\n📦 RESOURCE ALERT GENERATED:")
    print(f"Total Cost: ₹{resource_alert['total_cost']:,.2f}")
    print(f"Categories: {', '.join(resource_alert['resource_categories'])}")
    
    # Demo: Send follow-up
    follow_up = alert_system.send_follow_up_alerts(
        flood_alert['alert_id'],
        "Evacuation in progress. 500 people moved to safety. Relief camps operational."
    )
    
    print(f"\n📢 FOLLOW-UP SENT:")
    print(f"Follow-up ID: {follow_up['follow_up_id']}")
    
    # Demo: Alert statistics
    stats = alert_system.generate_alert_statistics()
    
    print(f"\n📊 ALERT STATISTICS:")
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"SMS Sent: {stats['total_sms_sent']}")
    print(f"Push Notifications: {stats['total_push_notifications']}")
    
    print(f"\n🎉 Alert system demonstration completed!")

if __name__ == "__main__":
    main()