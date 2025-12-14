import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Section1 from './components/Section1.jsx'
import Section2 from './components/Section2.jsx'
import Topbar from './components/TopBar.jsx'
import {motion} from 'framer-motion'
function Home() {
  const [count, setCount] = useState(0)

  return (
    <>

        <div className='contain'
        >
          <Topbar />
          <motion.div
            initial={{ opacity: 0, filter: "blur(10px)" }}
            animate={{ opacity: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0, filter: "blur(10px)" }}
            transition={{ duration: 0.6, ease: "easeOut" }}
    >
            <Section1 />
            {/* <Section2 /> */}
            </motion.div>
        </div>

    </>
  )
}

export default Home
