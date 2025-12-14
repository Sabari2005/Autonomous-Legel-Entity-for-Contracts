// import { useState, useEffect, useContext } from "react";
// import { Layout, Menu, Drawer, Button } from "antd";
// import { SyncOutlined } from "@ant-design/icons";
// import { useNavigate } from "react-router";
// import Notification from "./Notification";
// import { 
//   HomeOutlined, 
//   UserOutlined, 
//   SettingOutlined, 
//   MenuOutlined,
//   AuditOutlined ,
//   DatabaseOutlined,
//   ZoomOutOutlined,
//   MonitorOutlined
// } from "@ant-design/icons";
// import { Scale } from "lucide-react";
// import {ExtractedTextContext} from "../ExtractedTextContext"; // Adjust the import path as necessary

// const { Sider } = Layout;

// const Sidebar = () => {
// const navigate = useNavigate();
//   const [isOpen, setIsOpen] = useState(false);
//   const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
//   const [drawerOpen, setDrawerOpen] = useState(false);
//   const { SidebarOperation } = useContext(ExtractedTextContext);
//   const [showNotification, setShowNotification] = useState(false);

//   useEffect(() => {
//     if (SidebarOperation) {
//       setShowNotification(true);
  
//       const timer = setTimeout(() => {
//         setShowNotification(false);
//       }, 3000); // ⏰ Hide after 3 seconds
  
//       return () => clearTimeout(timer); // Cleanup on unmount or re-trigger
//     }
//   }, [SidebarOperation]); // 🔁 Runs whenever SidebarOperation changes
//   // Handle window resize
//   useEffect(() => {
//     const handleResize = () => {
//       const mobile = window.innerWidth <= 768;
//       setIsMobile(mobile);
//       setIsOpen(!mobile);
//     };

//     window.addEventListener("resize", handleResize);
//     return () => window.removeEventListener("resize", handleResize);
//   }, []);

//   return (
//     <>

//       {/* Desktop Sidebar */}
//       {!isMobile && (
//         <Sider
//           collapsible
//           collapsed={!isOpen}
//           onMouseEnter={() => {
 
//               setIsOpen(true);
            
//           }}
//           onMouseLeave={() => {

//               setIsOpen(false);
          
//           }}
//           trigger={null}
//           background="#F5FBFF"
//           width={250}
//           style={{
//             position: "absolute",
//             top: 0,
//             left: 0,
//             height: "100vh",
//             background: "#F5FBFF",
//             borderRight: "1px solid #ddd",
//             zIndex: 1000,
//             boxShadow: "2px 0 5px rgba(0, 0, 0, 0.1)",
//             pointerEvents: SidebarOperation ? "none" : "auto",
//           }}
//         >
//           <div
//             style={{
//               padding: "16px",
//               textAlign: "center",
//               fontWeight: "bold",
//               fontSize: "18px",
//               borderBottom: "1px solid #ddd",
//             }}
//           >
//             {isOpen ? 
//             <div style={{
//                 display:"flex",
//                 alignItems:"center",
//                 gap:"10px",
//                 // background:"yellow"
//               }}>
//                 <div style={{
//                     height:50,
//                     width:50,
//                     background:"#F5FBFF",
//                     padding:5,
//                     borderRadius:20

//                 }}>
//                     <div style={{
//                         height:40,
//                         width:40,
//                         background:"#0e1c29",
//                         padding:8,
//                         borderRadius:15

//                     }}>
//                         <img src="./logowhite.svg" style={{height:"100%",width:"100%", objectFit:"contain"}} ></img>
//                     </div>

//                 </div>  
//                 <span>Alec</span>  
//             </div>        
//             : 
//             <div style={{
//                 height:50,
//                 width:50,
//                 background:"#E1F6F6",
//                 padding:5,
//                 borderRadius:20}}>
//                 <div style={{
//                     height:40,
//                     width:40,
//                     background:"#0e1c29",
//                     padding:8,
//                     borderRadius:15}}>
//                     <img src="./logowhite.svg" style={{height:"100%",width:"100%", objectFit:"contain"}} ></img>
//                 </div>

//               </div> 
//             }
//           </div>

//           {/* Top Menu Items */}
//           <Menu theme="light" mode="inline" style={{
//     background: "#F5FBFF", // Set the background color
//     borderRight: "none",
//   }}>
//             <Menu.Item key="1" icon={<HomeOutlined style={{paddingLeft: isOpen ? "2px" : "0px",scale:1.1}}  />}
//             onClick={() => navigate("/")}>Home</Menu.Item>
//             <Menu.Item key="2" icon={<img src="./star.svg" style={{
//                 width: 20,
//                 height: 20,
//                 transition: "transform 0.3s ease-in-out",
//                 transform: isOpen ? "scale(1.3)" : "scale(1.3)",
//               }} />}onClick={() => navigate("/AiC")}>AI Contracter</Menu.Item>
//             {/* <Menu.Item key="3" icon={<img src="./search.svg" style={{
//                 width: 20,
//                 height: 20,
//                 transition: "transform 0.3s ease-in-out",
//                 transform: "scale(1.3)",
//               }}/>} onClick={() => navigate("/RSk")}>Risk Analyser</Menu.Item> */}
//               <Menu.Item key="3" icon={<ZoomOutOutlined />} onClick={() => navigate("/RSk")}>Risk Analyser</Menu.Item>
//             <Menu.Item key="4" icon={<SyncOutlined />} onClick={() => navigate("/Smz")}>Summerrizer</Menu.Item>
//             <Menu.Item key="5" icon={<AuditOutlined />} onClick={() => navigate("/Cde")}>
//               Company Deligence
//             </Menu.Item>
//             {/* <Menu.Item key="6" icon={<img src="./server.svg" style={{
//                 width: 20,
//                 height: 20,
//                 transition: "transform 0.3s ease-in-out",
//                 transform: isOpen ? "scale(1.3)" : "scale(1.3)",
//             }}/>} onClick={() => navigate("/Fs")}>Storage</Menu.Item> */}
//             <Menu.Item key="6" icon={<DatabaseOutlined />}>Storage</Menu.Item>
            

//           </Menu>

//           {/* Bottom Profile & Settings */}
//           <Menu
//             theme="light"
//             mode="inline"
//             style={{

//     background: "#F5FBFF", // Set the background color
//     borderRight: "none",

//               position: "absolute",
//               bottom: 20,
//               width: "100%",
//             }}
//           >
//             <Menu.Item key="7" icon={<UserOutlined />}onClick={() => navigate("/Acct")}>Profile</Menu.Item>
//             <Menu.Item key="8" icon={<SettingOutlined />}onClick={() => navigate("/Sts")}>Settings</Menu.Item>
//           </Menu>
//         </Sider>
//       )}

//       {/* Mobile Topbar */}
//       {isMobile && (
//         <div
//           style={{
//             position: "absolute",
//             top: 0,
//             left: 0,
//             width: "100%",
//             height: "60px",
//             background: "#F5FBFF",
//             boxShadow: "0 2px 5px rgba(0, 0, 0, 0.1)",
//             display: "flex",
//             alignItems: "center",
//             padding: "0 16px",
//             zIndex: 1100,
//             pointerEvents: SidebarOperation ? "none" : "auto",
//           }}
//         >
//           <Button
//             icon={<MenuOutlined />}
//             onClick={() => setDrawerOpen(true)}
//             style={{
//               fontSize: "20px",
//               background: "none",
//               border: "none",
//               cursor: "pointer",
//             }}
//           />
//           <div style={{ marginLeft: "16px", fontSize: "18px", fontWeight: "bold" }}>
//             Alec
//           </div>
//         </div>
//       )}

//       {/* Mobile Drawer Menu */}
//       <Drawer
//         title="Menu"
//         placement="left"
//         closable
//         onClose={() => setDrawerOpen(false)}
//         open={drawerOpen} 
//         style={{
//     background: "#F5FBFF", // Set the background color
//     borderRight: "none",
//   }}
//       >
//         <Menu theme="light" mode="vertical" style={{
//     background: "#F5FBFF", // Set the background color
//     borderRight: "none",
//   }}>
//           <Menu.Item key="1" icon={<HomeOutlined />}>Home</Menu.Item>
//           {/* <Menu.Item key="2" icon={<img src="./search.svg" style={{
//                 width: 20,
//                 height: 20,
//                 transition: "transform 0.3s ease-in-out",
//                 transform: isOpen ? "scale(1.3)" : "scale(1.4)",
//               }}/>} onClick={() => navigate("/RSk")}>Risk Analyser</Menu.Item> */}

// <Menu.Item key="3" icon={<ZoomOutOutlined />} onClick={() => navigate("/RSk")}>Risk Analyser</Menu.Item>
//             <Menu.Item key="3" icon={<img src="./star.svg" style={{
//                 width: 20,
//                 height: 20,
//                 transition: "transform 0.3s ease-in-out",
//                 transform: isOpen ? "scale(1.3)" : "scale(1.4)",
//               }}/>} onClick={() => navigate("/AiC")}>AI Contracter</Menu.Item>
//             <Menu.Item key="4" icon={<img src="./server.svg" style={{
//                 width: 20,
//                 height: 20,
//                 transition: "transform 0.3s ease-in-out",
//                 transform: isOpen ? "scale(1.2)" : "scale(1.2)",
//             }}/>} onClick={() => navigate("/Fs")}>Storage</Menu.Item>
//             <Menu.Item key="5" icon={<AuditOutlined />} onClick={() => navigate("/Cde")}>
//               Company_Deligence
//             </Menu.Item>
//             <Menu.Item key="6" icon={<SyncOutlined />} onClick={() => navigate("/Process")}>
//               Process
//             </Menu.Item>

//           <Menu.Item key="7" icon={<UserOutlined />} onClick={() => navigate("/Acct")}>Profile</Menu.Item>
//           <Menu.Item key="8" icon={<SettingOutlined />} onClick={() => navigate("/Sts")}>Settings</Menu.Item>
//         </Menu>
//       </Drawer>


//     </>
//   );
// };

// export default Sidebar;
import { useState, useEffect, useContext } from "react";
import { Layout, Menu } from "antd";
import { SyncOutlined } from "@ant-design/icons";
import { useNavigate } from "react-router";
import {
  HomeOutlined,
  UserOutlined,
  SettingOutlined,
  AuditOutlined,
  DatabaseOutlined,
  ZoomOutOutlined,
} from "@ant-design/icons";
import { ExtractedTextContext } from "../ExtractedTextContext"; // Adjust path if needed

const { Sider } = Layout;

const Sidebar = () => {
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false); // Always open on desktop
  const { SidebarOperation } = useContext(ExtractedTextContext);
  const [showNotification, setShowNotification] = useState(false);

  useEffect(() => {
    if (SidebarOperation) {
      setShowNotification(true);

      const timer = setTimeout(() => {
        setShowNotification(false);
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [SidebarOperation]);

  return (
    <Sider
      collapsible
      collapsed={!isOpen}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
      trigger={null}
      width={250}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        height: "100vh",
        background: "#F5FBFF",
        borderRight: "1px solid #ddd",
        zIndex: 1000,
        boxShadow: "2px 0 5px rgba(0, 0, 0, 0.1)",
        pointerEvents: SidebarOperation ? "none" : "auto",
      }}
    >
      <div
        style={{
          padding: "16px",
          textAlign: "center",
          fontWeight: "bold",
          fontSize: "18px",
          borderBottom: "1px solid #ddd",
        }}
      >
        {isOpen ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <div
              style={{
                height: 50,
                width: 50,
                background: "#F5FBFF",
                padding: 5,
                borderRadius: 20,
              }}
            >
              <div
                style={{
                  height: 40,
                  width: 40,
                  background: "#0e1c29",
                  padding: 8,
                  borderRadius: 15,
                }}
              >
                <img
                  src="./logowhite.svg"
                  style={{
                    height: "100%",
                    width: "100%",
                    objectFit: "contain",
                  }}
                />
              </div>
            </div>
            <span>Alec</span>
          </div>
        ) : (
          <div
            style={{
              height: 50,
              width: 50,
              background: "#E1F6F6",
              padding: 5,
              borderRadius: 20,
            }}
          >
            <div
              style={{
                height: 40,
                width: 40,
                background: "#0e1c29",
                padding: 8,
                borderRadius: 15,
              }}
            >
              <img
                src="./logowhite.svg"
                style={{
                  height: "100%",
                  width: "100%",
                  objectFit: "contain",
                }}
              />
            </div>
          </div>
        )}
      </div>

      <Menu
        theme="light"
        mode="inline"
        style={{
          background: "#F5FBFF",
          borderRight: "none",
        }}
      >
        <Menu.Item
          key="1"
          icon={<HomeOutlined style={{ paddingLeft: isOpen ? "2px" : "0px", scale: 1.1 }} />}
          onClick={() => navigate("/")}
        >
          Home
        </Menu.Item>
        <Menu.Item
          key="2"
          icon={
            <img
              src="./star.svg"
              style={{
                width: 20,
                height: 20,
                transform: "scale(1.3)",
              }}
            />
          }
          onClick={() => navigate("/AiC")}
        >
          AI Contracter
        </Menu.Item>
        <Menu.Item key="3" icon={<ZoomOutOutlined />} onClick={() => navigate("/RSk")}>
          Risk Analyzer
        </Menu.Item>
        <Menu.Item key="4" icon={<SyncOutlined />} onClick={() => navigate("/Smz")}>
          Summarizer
        </Menu.Item>
        <Menu.Item key="5" icon={<AuditOutlined />} onClick={() => navigate("/Cde")}>
          Company Deligence
        </Menu.Item>
        <Menu.Item key="6" icon={<DatabaseOutlined />} onClick={() => navigate("/Fs")}>
          Storage
        </Menu.Item>
      </Menu>

      <Menu
        theme="light"
        mode="inline"
        style={{
          background: "#F5FBFF",
          borderRight: "none",
          position: "absolute",
          bottom: 20,
          width: "100%",
        }}
      >
        <Menu.Item key="7" icon={<UserOutlined />} onClick={() => navigate("/Acct")}>
          Profile
        </Menu.Item>
        <Menu.Item key="8" icon={<SettingOutlined />} onClick={() => navigate("/Sts")}>
          Settings
        </Menu.Item>
      </Menu>
    </Sider>
  );
};

export default Sidebar;

